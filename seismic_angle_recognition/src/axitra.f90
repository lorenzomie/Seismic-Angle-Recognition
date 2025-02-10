!******************************************************************************
!*             AXITRA Moment Version
!
!*             PROGRAMME AXITRA
!*
!*           Calcul de sismogrammes synthetiques en milieu stratifie a symetrie
!*      cylindrique.
!*        Propagation par la methode de la reflectivite, avec coordonnees
!*      cylindriques (r, theta, z)
!*      Attenuation sur les ondes P et S
!*
!*      auteur : Olivier Coutant
!*        Bibliographie :
!*                      Kennett GJRAS vol57, pp557R, 1979
!*                        Bouchon JGR vol71, n4, pp959, 1981
!*
!******************************************************************************

program axitra

   use dimension1
   use dimension2
   use parameter
   use initdatam
   use reflect0m
   use reflect1m
   use reflect2m
   use reflect3m
   use reflect4m
   use reflect5m
   use allocatearraym

! the following works with intel compiler
! it may be necessary to remove it and explicitely declare instead
! integer :: omp_get_num_threads, omp_get_thread_num
#if defined(_OPENMP)
   use omp_lib
#endif

   implicit none
! Local
   character(len=20)    :: sourcefile, statfile, header,arg
   integer              :: ic, ir, is, nfreq, ikmax, ncp, iklast, jf, ik, lastik
   integer              :: nrs ! number of receiver radial distance
   integer              :: ncr ! number of layer containing a receiver
   integer              :: ncs ! number of layer containing a source

   integer              :: nr, ns, nc, narg
   real(kind=fd)         :: dfreq, freq, pil
   logical              :: latlon, freesurface,uflow1,uflow3,uflow4
   logical, allocatable :: tconv(:, :)
   real(kind=fd)         :: rw, aw, phi, zom, tl, xl, rmax, vlim, vmean
   namelist/input/nfreq, tl, aw, xl, ikmax, latlon, freesurface, sourcefile, statfile

   uflow1=.false.
   uflow3=.false.
   uflow4=.false.

#include "version.h"
   write(0,*) 'running axitra '//VERSION

!
! read header if any
!
   narg=iargc()
   if (narg>0) then
     call getarg(1,arg)
     write(header,"('axi_',A)") trim(arg)
   else
     header='axi'
   endif
!++++++++++
!           Read input parameter from <header>.data
!           <header>.head is used later to know the exact number of frequency
!           actually computed
!
!++++++++++
   open (in1, form='formatted', file=trim(header)//'.data')
   open (out, form='formatted', file=trim(header)//'.head')

! count number of layer
   read (in1, input)
   nc=0
   do while(.true.)
     read(in1,*,end=91)
     nc=nc+1
   end do
91 rewind(in1)
   read (in1, input)

   open (in2, form='formatted', file=sourcefile)
   open (in3, form='formatted', file=statfile)

! count number of sources
   ns=0
   do while(.true.)
      read(in2,*,end=92)
      ns=ns+1
   end do
92 rewind(in2)
! count number of receiver
   nr=0
   do while(.true.)
     read(in3,*,end=93)
     nr=nr+1
   end do
93 rewind(in3)

   write(6,*) ' with ',ns,'source(s) and ',nr,'receiver(s) and ',nc,'layer(s)'
   if (freesurface) then
      write(6,*) '................. with a free surface at depth Z=0'
   else
      write(6,*) '................. with no free surface'
   endif

   call allocateArray1(nc, nr, ns)

!
! read velocity model
!
   do ic = 1, nc
      read (in1, *) hc(ic), vp(ic), vs(ic), rho(ic), qp(ic), qs(ic)
   enddo

!
! check unit, m or km?
! if average seismic velocities are < 20 (no dimension number)
! assume km/s velocities
!
   vmean=(mean(vp)+mean(vs))/2.d0
   if (vmean<20.d0) then ! assume velocities are in km / sec
      write(6,*) 'distance unit is detected as Kilometer (average velocity values < 20)'
      if (mean(rho) > 1) then
         write(6,*) 'Inconsistent density unit, must be Kg/km^3 and be of the order of 10-6'
         call exit(1)
      end if
   else
      write(6,*) 'distance unit is detected as meter'
      if (mean(rho) < 1) then
         write(6,*) 'Inconsistent density unit, must be Kg/m^3 and be of the order of 10+3'
         call exit(1)
      end if
   endif
   if (latlon .and. vmean<20.d0) then
      write(6,*) 'inconsistent velocity units when coordinates are in latitude/longitude'
      write(6,*) 'expect velocity as meter/sec and depth in meter'
      call exit(1)
   endif


! We assume here that record length is given in byte.
! For intel compiler, it means using "assume byterecl" option
! record length is 6 x (complex double precision) = 6 x 2 x 8 bytes
   open (out2, access='direct', recl=6*3*2*8*nr*ns, form='unformatted',file=trim(header)//'.res')
! these additionnal options may help increasing IO when using ifort
!        buffered='yes',buffercount=24)

!
!++++++++++
!           INITIALISATIONS
!++++++++++

   call initdata(latlon, nr, ns, nc, ncr, ncs,nrs,rmax)

! compute xl and or tl if not supplied
   if (xl<=0.d0 .or. tl<=0.d0) call estimateParameter(xl,tl,rmax,vp,vs,nc)
   write(out,*) xl,tl

   allocate (jj0(nkmax, nrs))
   allocate (jj1(nkmax, nrs))

   uconv = rerr*rerr
   dfreq = 1./tl
   aw = -pi*aw/tl
   pil = pi2/xl
   iklast = 0

!$OMP PARALLEL DEFAULT(FIRSTPRIVATE) &
!$OMP SHARED(dfreq,iklast,nc,nr,ns,ncs,ncr,uconv) &
!$OMP SHARED(tl,aw,pil,cff,jj0,jj1)
#if defined(_OPENMP)
   if (omp_get_thread_num()==1) then
       write(0,*) 'running openMp on ',omp_get_num_threads(),' threads'
   endif
#endif
   call allocateArray2(nc, nr, ns,nrs)
   allocate (tconv(nr, ns))

!               ***************************
!               ***************************
!               **  BOUCLE EN FREQUENCE  **
!               ***************************
!               ***************************

!$OMP DO ORDERED,SCHEDULE(DYNAMIC)
   do jf = 1, nfreq

      freq = (jf - 1)*dfreq
!      write (6, *) 'freq', jf, '/', nfreq
      rw = pi2*freq
      omega = cmplx(rw, aw)
      omega2 = omega*omega
      a1 = .5/omega2/xl
      zom = sqrt(rw*rw + aw*aw)
      if (jf .eq. 1) then
         phi = -pi/2
      else
         phi = atan(aw/rw)
      endif
      do ir = 1, nr
         do is = 1, ns
            tconv(ir, is) = .false.
         enddo
      enddo

      ttconv = .false.
      xlnf = (ai*phi + dlog(zom))/pi
! Futterman
      xlnf = (ai*phi + dlog(zom/(pi2*fref)))
! Kjartansson
      xlnf = zom/(pi2*fref)

!            ******************************************
!            ******************************************
!            **  RESOLUTION PAR BOUCLE EXTERNE EN Kr **
!            ******************************************
!            ******************************************

      do ik = 0, ikmax

         kr = (ik + .258)*pil
         kr2 = kr*kr

!+++++++++++++
!              Calcul de nombreux coefficients et des fonctions de Bessel
!+++++++++++++

         call reflect0(ik + 1, iklast, nc, nr,ns, nrs)

!+++++++++++++
!              Calcul des coefficients de reflexion/transmission
!               Matrice de Reflection/Transmission et Dephasage
!+++++++++++++

         call reflect1(freesurface, nc, uflow1)

!+++++++++++++
!              Calcul des matrices de reflectivite : mt(),mb(),nt(),nb()
!              (rapport des differents potentiels montant/descendant
!                        en haut et en bas de chaque couche)
!+++++++++++++

         call reflect2(nc)

!+++++++++++++
!               Calcul des matrices de passage des vecteurs potentiel
!                source, aux vecteurs potentiel PHI, PSI et KHI au sommet
!                de chaque couche
!+++++++++++++
         call reflect3(ncs, uflow3)

!+++++++++++++
!               Calcul des potentiels et des deplacement dus aux sources du
!                tenseur, en chaque recepteur (termes en kr, r, z)
!+++++++++++++
         call reflect4(jf, ik, ik .gt. ikmin, tconv, nc, nr, ns, ncs, ncr, uflow4)

         if (ttconv) exit

      end do !wavenumber loop

!+++++++++++++
!               Calcul des deplacements aux recepteurs
!                Sortie des resultats
!+++++++++++++

      lastik = ik - 1
      write (out, *) 'freq =', freq, 'iter =', lastik
      write (6,"(1a1,'freq ',I5,'/',I5,' iter=',I9,$)") char(13),jf, nfreq,lastik

      if (jf .eq. 1) lastik = 0

      call reflect5(jf, nr, ns)

      if (ik .ge. ikmax) then
         write (0, *) 'Abort: reached max iteration number for frequency ',freq
         if (uflow4 .or. uflow1 .or. uflow3) then
            write(0,*) 'Since underflow occured, consider introducing a fictituous layer'
            write(0,*) 'to reduce the vertical distance between the top of the layer and '
            write(0,*) 'the source/receiver depth'
            write(0,*) 'uflow',uflow1,uflow3,uflow4
            call exit(2)
         else
            call exit(1)
         endif
      endif

   enddo !boucle freq
!$OMP END PARALLEL
   write(6,*) 'Done'
   call exit(0)
end

!
! estimate tl and/or xl parameter in case they were not given by user
!

subroutine estimateParameter(xl,tl,rmax,vp,vs,nc)
use parameter
implicit none
real(kind=fd) :: xl,tl,rmax,vp(nc),vs(nc)
integer      :: nc

integer      :: i
real(kind=fd) :: vmax,vmin

vmax=vp(1)
vmin=vs(1)
do i=2,nc
   vmax=max(vmax,vp(i))
   vmin=min(vmin,vs(i))
enddo

! estimate a duration
! we need at least the time needed to travel
! along the largest distance at the lowest velocity
vmin=0.8d0*vmin
if (tl<=0.d0) then
    tl=rmax/vmin
    tl=(int(tl/5.d0)+1)*5.d0
write(6,*) 'duration tl set automatically to ',tl,'sec'
endif

! estimate a radial periodicity
! source are located with a xl periodicity
! To avoid the effect of periodic sources during the 
! tl duration, we need:
if (xl<=0.d0) then
    xl = 1.2d0*rmax + vmax*tl
    xl=(int(xl/10.d0)+1)*10.d0
    write(6,*) 'periodicity xl set automatically to ',xl,'(k)m'
endif
end subroutine
