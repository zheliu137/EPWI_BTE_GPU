!!------------------------------------------------------------------------------
SUBROUTINE run_BTE()
!!------------------------------------------------------------------------------
  !
  USE global_variables
  USE newinputs
  USE global_constants
  !
  USE mp,            ONLY : mp_barrier, mp_bcast, mp_sum
  USE mp_global,     ONLY : inter_pool_comm, intra_pool_comm, my_pool_id, npool
  USE mp_world,      ONLY : mpime, nproc, world_comm
  USE epwcom,        ONLY : lpolar, lifc
  USE io_files,      ONLY : tmp_dir
  !
  IMPLICIT NONE
  integer(kind=LI) :: Ntotal
  integer(kind=4),allocatable :: N_plus(:),N_minus(:)
  integer :: i,j,k,ii,jj,kk,ll,mm,nn,Tcount
  real(kind=8),allocatable :: F_n(:,:,:),DeltaF(:,:,:)
  integer :: iconcent,Nconcent,ios
  integer :: f_mkdir_sf
  real(kind=8),allocatable :: concent_range(:)
  integer,allocatable :: Eqindex_K_tmp(:,:)
  real(kind=8),allocatable :: el_energy_tmp(:,:), el_velocity_tmp(:,:,:)
  integer :: vec(3),ind(2)
  character(len=6) :: aux
  character (len=256) :: filint, tmp_dir_nc
  character (len=3) :: filelab
  real(kind=8) :: concent, relchange
  real(kind=8) :: Elcond(3,3)
  real(kind=8), EXTERNAL :: dnrm2
  real(kind=8) :: mem_tmp
  ! ---------- GPU variables -------------
  !
  integer,ALLOCATABLE :: Eqindex_K_gpu(:,:), List_gpu_tmp(:), List_gpu(:)
  ! ---------- GPU variables -------------
  !
  CALL start_clock('BTE_prepare')
  !
  CALL prepare_lattice()
  !
  call read_config()
  !
  allocate(Eqindex_K(NPTK_K,2))
  allocate(List_gpu_tmp(NPTK_K))
  CALL wedge_gpu(Nlist_K, List_gpu_tmp, Eqindex_K, nsymm, NKgrid, Orth_rev, symm_rev_idx)
  allocate(List_K(Nlist_K))
  List_K=List_gpu_tmp(1:Nlist_K)
  deallocate(List_gpu_tmp)
  allocate(el_energy(Nlist_K,Nbands),el_velocity(Nlist_K,Nbands,3))
  call band_wannier_gpu(Nlist_K, List_K, el_energy, el_velocity)
  !
  !
  if (type_key.eq."type_N") then
      el_energy=el_energy-minval(el_energy)
  elseif (type_key.eq."type_P") then
      el_energy=el_energy-maxval(el_energy)
  endif
  if (type_key.eq."type_M") then
      el_energy=el_energy-Efermi  ! in unit of ev
  endif
  !
  !! phonon calculation
  CALL start_clock("BTE_phdisp_vq")
  allocate(Eqindex_Q(NPTK_Q,2))
  allocate(List_gpu_tmp(NPTK_Q))
  CALL wedge_gpu(Nlist_Q, List_gpu_tmp, Eqindex_Q, nsymm, NQgrid, Orth_rev, symm_rev_idx)
  allocate(List_Q(Nlist_Q))
  List_Q=List_gpu_tmp(1:Nlist_Q)
  deallocate(List_gpu_tmp)
  if (mpime.eq.0) write(*,'("Info: Number of phonons",3I6,I12)') NQgrid, Nlist_Q
  allocate (ph_energy(Nlist_Q,Nmodes),ph_velocity(Nlist_Q,Nmodes,3))
  call phonon_wannier_gpu(Nlist_Q, List_Q, ph_energy, ph_velocity)
  CALL stop_clock("BTE_phdisp_vq")
  CALL stop_clock('BTE_prepare')
  !
  !
  if (mpime.eq.0) write(*,*) "Info:   electron_phonon calculation  "
  CALL start_clock('BTE_elph_Nprocess')
  !
  call ElectronInterest(Nlist_K,Ethreshold)
  !
  allocate( N_plus(NStateInterest) )
  allocate( naccum(NStates+1) )
  allocate( PhononInterest(NPTK_Q) )
  
  PhononInterest=0
  N_plus=0
  naccum=0

  CALL gpu_cuda_init_wrap()
  call Nprocesses_cuda(N_plus, naccum, PhononInterest)
  call Nprocesses_cuda_deal()
  NPhononInterest=0

  do mm=1,NPTK_Q
      if (PhononInterest(mm).ne.0) then  !! for wannier interpolated g2element
          NPhononInterest=NPhononInterest+1
          PhononInterest(mm)=NPhononInterest !! for g2element 2nd-phonon index
      endif
  enddo 
  if (mpime.eq.0)  &
              write(*,'("Info: Number of kq-pairs : ",1000I10)') Ntotal 
  allocate(F_n(Nlist_K,Nbands,3),DeltaF(Nlist_K,Nbands,3))
  F_n=0.d0
  DeltaF=0.d0
  !! to reduce the memory, using parallel for Indscatt, Gamma, g2element and related calculation
  if (convergence) then
      allocate( Gamma(naccum(Nstates+1)), &
                indscatt(naccum(Nstates+1)) )
      mem_tmp = dble(Ntotal)*24.0/1024**3
      if (mpime.eq.0) & 
          write(*,'("Info: Estimated total memory needed by Gamma : ",f7.3 ," GB ")') mem_tmp
      Gamma=0.d0
  endif
  !
  if (mpime.eq.0) write(*,'("Info: El-Ph matrix wannier interpolation parallel started")')
  !
  CALL stop_clock('BTE_elph_Nprocess')
  !
  CALL start_clock('W2B_Core')
  !
  if (mpime.eq.0) write(*,'("Info: Temperature =",F10.2)') Te
  if (mpime.eq.0) write(*,'("Info: Fermi level =",F10.2)') ChemPot
  !
  CALL elphwann_wannier_gpu()
  !
  CALL cuda_elphwann_wannier_destroy()
  !
  CALL stop_clock('W2B_Core')
  !
  CALL start_clock('BTE_iter')
  ! open files for output datas
  tmp_dir_nc='output/'
  ios=f_mkdir_sf(trim(tmp_dir_nc))
  if (mpime.eq.0) then
      open(1001,file=trim(tmp_dir_nc)//'BTE.sigmavsT_RTA',status='replace')
      if (convergence) then
          open(2001,file=trim(tmp_dir_nc)//'BTE.sigmavsT_ITER',status='replace')
      endif
      open(3001,file=trim(tmp_dir_nc)//'BTE.sigmavsT_MRTA',status='replace')
  endif 
  ! output scattering rate
  if (mpime.eq.0) then
      write(aux,"(I0)") NINT(Te)
      open(1,file=trim(tmp_dir_nc)//'BTE.sr.T'//trim(adjustl(aux))//'K',status='replace')
      do mm=1,NStateInterest
          i=StateInterest(1,mm)
          ll=StateInterest(2,mm)
          write(1,"(I6,4E20.10)") i,el_energy(ll,i),rate_scatt(mm),rate_scatt_mrta(mm),&
                          dnrm2(3,velocity(ll,i,:),1)/rate_scatt(mm)/radps2ev
      enddo
      close(1)
  endif

  do mm=1,NStateInterest
      if(rate_scatt(mm).gt.0) then
          rate_scatt(mm)=1.d0/rate_scatt(mm)
      else
          rate_scatt(mm)=0.d0
      endif
  ENDDO
  !
  write(aux,"(I0)") 3*Nband  
  !
  do mm=1,NStateInterest
      i=StateInterest(1,mm)
      ll=StateInterest(2,mm)
      F_n(ll,i,:)=rate_scatt(mm)*velocity(ll,i,:)
  enddo
  !
  if (mpime.eq.0) then
      call ElConduct(F_n,Elcond)
      if (type_key.eq."type_M") then
          write(1001,"(F10.2,3E20.10)") Te, Elcond(1,1),Elcond(2,2),Elcond(3,3)
      else
          write(1001,"(F10.2,7E20.10,F10.5)") Te,&
              (/Elcond(1,1),Elcond(2,2),Elcond(3,3)/)/echarge*1.d-2/concent, & ! mobility in unit of cm^2/V-s
              Elcond(1,1),Elcond(2,2),Elcond(3,3),concent,ChemPot! carrier concentration in unit of 1/cm^3
      endif
  endif  
  !
  if(convergence) then
      if (mpime.eq.0) write(*,'("Info: Iteration start...")'  )
      CALL cuda_iter_init()
      CALL cuda_iter(rate_scatt, naccum, Elcond)
      CALL cuda_iter_destroy()
      if (type_key.eq."type_M") then
          write(2001,"(F10.2,3E20.10)") Te, Elcond(1,1),Elcond(2,2),Elcond(3,3)
      else
          write(2001,"(F10.2,7E20.10,F10.5)") Te,&
              (/Elcond(1,1),Elcond(2,2),Elcond(3,3)/)/echarge*1.d-2/concent, & ! mobility in unit of cm^2/V-s
              Elcond(1,1),Elcond(2,2),Elcond(3,3),concent,ChemPot ! carrier concentration in unit of 1/cm^3
      endif
  endif
  ! deallocate arrays and close files
  if (mpime.eq.0) then
      close(1001)
      if (convergence) then
          close(2001)
      endif
      close(3001)
  endif
  if (convergence) then
      deallocate(indscatt)
      deallocate(Gamma)
  endif
  deallocate(N_plus)
  deallocate(rate_scatt,rate_scatt_mrta,rate_scatt_ph)
  deallocate(F_n,DeltaF)
  deallocate(StateInterest,PhononInterest)

  CALL stop_clock('BTE_iter')

  deallocate(List_K,Eqindex_K,List_Q,Eqindex_Q)
  deallocate(el_energy,el_velocity,ph_energy,ph_velocity)

!!------------------------------------------------------------------------------
END SUBROUTINE run_BTE
!!------------------------------------------------------------------------------

!!------------------------------------------------------------------------------
SUBROUTINE elphwann_wannier_gpu
!!------------------------------------------------------------------------------
  !
  USE global_variables
  USE newinputs
  USE mp_world,             ONLY: mpime
  USE global_constants,  ONLY: LI

  IMPLICIT NONE

  INTEGER :: ind_state, ind_band1, ind_k 
  INTEGER :: iq, ik, ikk, ikq, qcount, is, ikis
  INTEGER :: nn, mm, i, ll
  INTEGER :: ibnd, jbnd, imode, na, mu, nu
  INTEGER :: xkk1, xkq1, xkk2, xkq2, xkk3, xkq3, ir
  REAL(kind=8) :: g2_tmp
  REAL(kind=8) :: g2(nmodes, nbands)
  COMPLEX(kind=8), ALLOCATABLE :: epmatf_gpu(:, :, :)
  COMPLEX(kind=8), ALLOCATABLE :: epmatf_pre_gpu(:, :, :)
  COMPLEX(kind=8), ALLOCATABLE :: epmatl_gpu(:, :, :)

  INTEGER :: vec(3)
  INTEGER :: N_plus_count(NStateInterest)
  REAL(kind=8) :: etkk(nbndsub), etkq(nbndsub), wfqq(nmodes)
  INTEGER :: batchsize_k
  INTEGER :: batchsize_
  INTEGER :: batchsize_k_
  INTEGER, ALLOCATABLE :: valid_list(:)
  INTEGER :: nvalid
  INTEGER :: batch_count, batch_id, b_idx, exceed = 0
  COMPLEX(kind=DP) :: cfac_batched(nrr_k, batchSize), cfacq_batched(nrr_k, batchSize)
  REAL(kind=8) :: xkk_batched(3, batchSize)
  ! debug vari
  REAL(kind=8), ALLOCATABLE :: rate_scatt_(:), rate_scatt_mrta_(:)
  INTEGER(LI) :: gm_offset

  COMPLEX(kind=DP) :: umn(Nbands, 1)
  COMPLEX(kind=DP), ALLOCATABLE :: cufkk_gpu(:, :, :), cufkq_gpu(:, :, :), umn_gpu(:, :, :), uf_gpu(:, :, :), uf_dyn_gpu(:,:,:)
  REAL(kind=8), ALLOCATABLE :: g2_gpu(:,:,:), wq_gpu(:,:)
  INTEGER :: qveci(3)

  batchsize_k = batchsize*10

  ALLOCATE (rate_scatt(NStateInterest))
  ALLOCATE (rate_scatt_mrta(NStateInterest))
  ALLOCATE (rate_scatt_ph(NStateInterest, Nmodes, 3))
  ALLOCATE (rate_phase_ph(NStateInterest, Nmodes, 4))
  ALLOCATE (valid_list(NPTK_Q))
  ALLOCATE (rate_scatt_(NStateInterest))
  ALLOCATE (rate_scatt_mrta_(NStateInterest))

  rate_scatt_ = 0.D0
  rate_scatt_mrta_ = 0.D0
  rate_scatt = 0.D0
  rate_scatt_mrta = 0.D0
  rate_scatt_ph = 0.D0
  rate_phase_ph = 0.D0
  N_plus_count = 0
  gm_offset = 0
  CALL start_clock('GPU_init')

  batch_count = (Nstates + batchSize - 1)/batchSize

  CALL cuda_elphwann_restart(Nbands_irrelevant, ChemPot, el_velocity, ph_velocity, Gamma, indscatt)

  CALL stop_clock('GPU_init')

  DO is = 1, Nstates ! nstates
    !
    IF (mpime .EQ. 0 .AND. MOD(is, 100) .EQ. 1) &
      WRITE (*, '(6X,"Wannier interpolation",I10," of",I10," electron states k.")') is, Nstates
    !
    CALL start_clock('ephW2Be1')
    !
    ikis = StateInterest(2, is)
    ik = List_K(ikis)
    ibnd = StateInterest(1, is)
    !
    ! ------------------------------------------------------
    ! el Ham matrix : Wannier -> Bloch
    ! ------------------------------------------------------
    !
    CALL start_clock('Ham_1')
    IF (MOD(is, batchsize_k) == 1) THEN
      batch_id = is/batchsize_k
      !
      CALL cuda_hamwan2bloch(is - 1, batch_id)
      !
    END IF

    CALL stop_clock('Ham_1')
    !
    CALL start_clock('ephW2Be1_core')
    CALL cuda_ephwan2bloche(is - 1, ibnd)
    CALL stop_clock('ephW2Be1_core')
    !
    ! --------------------------------------------------------------
    ! epmat : Wannier el and Wannier ph -> Wannier el and Bloch ph
    ! --------------------------------------------------------------
    !
    CALL start_clock('CK_VALID')
    CALL check_valid_q(valid_list, is - 1, nvalid)
    CALL stop_clock('CK_VALID')
    !
    CALL stop_clock('ephW2Be1')
    !
    batch_count = (nvalid + batchSize - 1)/batchSize
    !
    DO batch_id = 1, batch_count ! q loop
      !
      !
      batchSize_ = MIN(nvalid - (batch_id - 1)*batchSize, batchSize)
      !
      CALL start_clock('ephW2Bep')
      !
      ! ------------------------------------------------------
      ! H_kq : Wannier -> Bloch
      ! eph R_p -> q FT and U_kq
      ! ------------------------------------------------------
      !
      CALL start_clock('W2B_hame2')
      CALL cuda_hamw2b_kq(batch_id - 1, ik - 1)
      CALL stop_clock('W2B_hame2')
      !
      ! ------------------------------------------------------
      ! Dyn Wannier -> Bloch
      ! eph U_ph
      ! ------------------------------------------------------
      !
      CALL start_clock('W2B_dyn')
      CALL cuda_dynwan2bloch_batched(batch_id - 1, eps)
      CALL stop_clock('W2B_dyn')
      !
      CALL stop_clock('ephW2Bep')
      !
      !
      IF (lpolar) THEN
        CALL cuda_polar(batch_id - 1, eps)
      END IF
      !
      CALL start_clock('Gamma')
      CALL cuda_g2(batch_id - 1)
      !
      CALL cuda_indpro(batch_id - 1, rate_scatt(is), rate_scatt_mrta(is), gm_offset)
      !
      CALL stop_clock('Gamma')
      !
    END DO ! end loop over q points
    !
  END DO ! end loop over k points

  WRITE (*, '(6X, "EPWI_GPU done.")')

  CALL mp_sum(rate_scatt, inter_pool_comm)
  CALL mp_sum(rate_scatt_mrta, inter_pool_comm)
  CALL mp_sum(rate_scatt_ph, inter_pool_comm)
  CALL mp_sum(rate_phase_ph, inter_pool_comm)
  !
  CALL mp_barrier(inter_pool_comm)
  !
  CALL system_mem_usage(valueRSS)
  !
  WRITE (stdout, '(a)') '     ==================================================================='
  WRITE (stdout, '(a,i10,a)') '     Memory usage:  VmHWM =', valueRSS(2)/1024, 'Mb'
  WRITE (stdout, '(a,i10,a)') '                   VmPeak =', valueRSS(1)/1024, 'Mb'
  WRITE (stdout, '(a)') '     ==================================================================='
  WRITE (stdout, '(a)') '     '
  !
  !
END SUBROUTINE elphwann_wannier_gpu
!!------------------------------------------------------------------------------

!!!------------------------------------------------------------------------------
SUBROUTINE band_wannier_gpu(Nlist, List, energy, velocity)

  USE global_variables
  USE newinputs
  USE global_constants
  USE noncollin_module, ONLY: noncolin

  IMPLICIT NONE

  INTEGER :: NList, List(Nlist)
  REAL(kind=8) :: energy(Nlist, Nbands), velocity(Nlist, Nbands, 3), velocity_tmp(Nlist, Nbands, 3)

  INTEGER :: ik, ikk, lower_bnd, upper_bnd, ibnd
  INTEGER :: vec(3)
  REAL(kind=8) :: etkk(nbndsub), etkk_ks(nbndsub)
  COMPLEX(kind=8) :: dmekk(3, nbndsub, nbndsub)
  REAL(kind=8) ::  vmef_diag(3, nbndsub)
  COMPLEX(kind=8), ALLOCATABLE :: cufkk_gpu(:, :, :)

  REAL(kind=8) :: etkk_gpu(nbndsub, NList)
  REAL(kind=8) :: el_vel_gpu(nbndsub, 3, NList)
  REAL(kind=8) :: ryd2nmev
  REAL(kind=8) :: emax, emin
  LOGICAL :: lmetal = .FALSE.

  CALL fkbounds(Nlist, lower_bnd, upper_bnd)

  energy = 0.D0
  velocity = 0.D0
  ryd2nmev = ryd2ms1*hbar_js/echarge*1E9
  ! print*,'ryd2nmev = ', ryd2nmev
  ! 1 ryd = 1.0938457e6 m/s, 1 m/s = 1/hbar*J*m, hbar = h/twopi = 6.62e-34 j*s/twopi, 1J = 1/echarge eV
  ALLOCATE (cufkk_gpu(nbndsub, nbndsub, NList))

  PRINT *, "Wannier interpolation of electronic bands are calculated by GPU."

  CALL cuda_band_wannier_init(nbndsub, Nbands, Nbands_irrelevant, nrr_k, Nlist, List, NKgrid, irvec_r, ndegen_k, &
                              chw, batchsize, at, alat, rlattvec, eqindex_k, orthcar, nsymm)
  CALL cuda_band_wannier(etkk_gpu, el_vel_gpu)

  DO ik = lower_bnd, upper_bnd !!1, nkf
    energy(ik, :) = etkk_gpu(Nbands_irrelevant + 1:Nbands_irrelevant + Nbands, ik)*ryd2ev
    velocity(ik, :, :) = el_vel_gpu(Nbands_irrelevant + 1:Nbands_irrelevant + Nbands, :, ik)*ryd2nmev
  END DO !ik
  !
  !
  CALL mp_sum(energy, inter_pool_comm)

  CALL mp_sum(velocity, inter_pool_comm)

  CALL mp_barrier(inter_pool_comm)

  IF (type_key .EQ. "type_M") lmetal = .TRUE.

  CALL cuda_band_dos(Efermi, delta_shift_EFermi, nelec, Te, ismear_ecp, degauss, scalebroad, lmetal, DosFermi, V2Fermi)

  CALL cuda_band_wannier_destroy()

END SUBROUTINE band_wannier_gpu

!!------------------------------------------------------------------------------

!!------------------------------------------------------------------------------
SUBROUTINE phonon_wannier_gpu(Nlist, List, energy, velocity)
!!------------------------------------------------------------------------------

  USE global_variables
  USE newinputs
  USE global_constants
  IMPLICIT NONE

  INTEGER :: NList, List(Nlist)
  REAL(kind=8) :: energy(Nlist, nmodes), velocity(Nlist, nmodes, 3), velocity_tmp(Nlist, nmodes, 3)

  INTEGER :: iq, iqq, nu, lower_bnd, upper_bnd, ibnd

  INTEGER :: vec(3)

  REAL(kind=8) :: etkk(nmodes)
  COMPLEX(kind=8) :: dmekk(3, nmodes, nmodes)
  REAL(kind=8) ::  vmef_diag(3, nmodes)
  COMPLEX(kind=8), ALLOCATABLE :: cufq_gpu(:, :, :)

  REAL(kind=8) :: omega_gpu(nmodes, NList)
  REAL(kind=8) :: ph_vel_gpu(nmodes, 3, NList)
  REAL(kind=8) :: ryd2nmev
  !
  CALL fkbounds(Nlist, lower_bnd, upper_bnd)

  energy = 0.D0
  velocity = 0.D0
  ryd2nmev = ryd2ms1*hbar_js/echarge*1E9
  ! print*,'ryd2nmev = ', ryd2nmev
  ! 1 ryd = 1.0938457e6 m/s, 1 m/s = 1/hbar*J*m, hbar = h/twopi = 6.62e-34 j*s/twopi, 1J = 1/echarge eV
  ! allocate(cufq_gpu(nmodes, nmodes, NList))

  PRINT *, "Wannier interpolation of phonon dispersion are calculated by GPU."

  CALL cuda_ph_wannier_init(nmodes, nrr_q, Nlist, List, NQgrid, irvec_r, ndegen_q, rdw, &
                            batchsize*100, at, bg, alat, vol, nat, amass, ityp, tau, &
                            lpolar, zstar, epsi, nq1, nq2, nq3)

  CALL cuda_ph_wannier(omega_gpu, ph_vel_gpu)

  DO iq = lower_bnd, upper_bnd !!1, nkf
    energy(iq, :) = omega_gpu(:, iq)*ryd2ev/radps2ev
    ! energy(iq, :) = SQRT(ABS(omega_gpu(:, iq)))*ryd2ev/radps2ev
    velocity(iq, :, :) = ph_vel_gpu(:, :, iq)*ryd2nmev/radps2ev
  END DO !ik

  CALL cuda_ph_wannier_destroy()

  CALL mp_sum(energy, inter_pool_comm)

  CALL mp_sum(velocity, inter_pool_comm)

  CALL mp_barrier(inter_pool_comm)

!!------------------------------------------------------------------------------
END SUBROUTINE phonon_wannier_gpu
!!------------------------------------------------------------------------------

!!------------------------------------------------------------------------------
SUBROUTINE gpu_cuda_init_wrap()
!!------------------------------------------------------------------------------
  USE ephwann_interp
  USE allsymmetry, ONLY: Orthcar, Nsymm
  USE inputparameters, ONLY: batchSize, ismear_ecp, tolerance, maxiter, ph_cut, delta_mult
  USE elph_variables, ONLY: spin_degen
  CALL reconstruct_epmatwp()
  CALL gpu_cuda_init(Nbands, nbndsub, nrr_k, nrr_q, nmodes, alat, bg, rlattvec, Vol, &
                     nat, SIZE(amass), amass, ityp, spin_degen, tau, Nstates, NStateInterest, StateInterest, irvec_r, &
                     irvec, ndegen_k, ndegen_q, epmatwp, chw, rdw, lpolar, epsi, zstar, nq1, nq2, nq3, NKgrid, NQgrid, &
                     Nsymm, Nlist_K, Nlist_Q, NPTK_K, NPTK_Q, List_K, Eqindex_K, &
                     Eqindex_Q, Te_min, ismear_ecp, scalebroad, degauss, delta_mult, ph_cut, el_energy, ph_energy, &
                     el_velocity, ph_velocity, Orthcar, convergence, tolerance, maxiter, &
                     batchSize, mpime)
!!------------------------------------------------------------------------------
END SUBROUTINE
!!------------------------------------------------------------------------------

!!------------------------------------------------------------------------------
SUBROUTINE reconstruct_epmatwp
!!------------------------------------------------------------------------------
  USE ephwann_interp, ONLY: epmatwp, nbndsub, nmodes, nrr_q, nrr_k, nbndsub
  INTEGER :: im, irrq
  COMPLEX(kind=8), ALLOCATABLE :: epmatwp_(:, :, :, :, :)
  ALLOCATE (epmatwp_(nbndsub, nmodes, nrr_q, nbndsub, nrr_k))
  DO im = 1, nmodes
    DO irrq = 1, nrr_q
      epmatwp_(:, im, irrq, :, :) = epmatwp(:, :, :, im, irrq)
    END DO
  END DO
  epmatwp = epmatwp_
  DEALLOCATE (epmatwp_)
!!------------------------------------------------------------------------------
END SUBROUTINE reconstruct_epmatwp
!!------------------------------------------------------------------------------

!!------------------------------------------------------------------------------
SUBROUTINE ElConduct(F_n, Elcond)
!!------------------------------------------------------------------------------
    IMPLICIT NONE
    REAL(kind=8), INTENT(in) :: F_n(Nlist, Nbands, 3)
    REAL(kind=8), INTENT(out) :: Elcond(3, 3)
    REAL(kind=8) :: fFD, tmp(3, 3), etmp, vtmp(3), Ftmp(3)
    INTEGER(kind=4) :: ii, jj, dir1, dir
    Elcond = 0.D0
    Ninterm = 0.D0
    DO jj = 1, Nbands
        DO ii = 1, nptk
            etmp = energy(Eqindex(ii, 1), jj)
            vtmp = MATMUL(Orthcar(:, :, Eqindex(ii, 2)), velocity(Eqindex(ii, 1), jj, :))
            Ftmp = MATMUL(Orthcar(:, :, Eqindex(ii, 2)), F_n(Eqindex(ii, 1), jj, :))
            DO dir1 = 1, 3
                DO dir2 = 1, 3
                    tmp(dir1, dir2) = vtmp(dir1)*Ftmp(dir2)
                END DO
            END DO
            fFD = 1.D0/(EXP((etmp - ChemPot)/Kb/Te) + 1.D0)
            Elcond(:, :) = Elcond(:, :) + fFD*(1.D0 - fFD)*tmp
        END DO
    END DO
    Elcond = spin_degen*Elcond*echarge*1.D21/(radps2ev**2)/(Kb*Te*Vol*nptk)
    write(*,"(A,9ES20.10)")"elcond : ", Elcond
!!------------------------------------------------------------------------------
END SUBROUTINE ElConduct
!!------------------------------------------------------------------------------

!!------------------------------------------------------------------------------
subroutine ElectronInterest(Nlist_K,Ethreshold)
!!------------------------------------------------------------------------------
    USE elph_variables,   only: el_energy
    USE inputparameters,  only: Nbands
    USE states_interest,  only: StateInterest,NStateInterest,Nstates
    USE mp_world,         ONLY: mpime, nproc
    IMPLICIT NONE
    integer :: Nlist_K,iband,ii
    real(kind=8) :: Ethreshold

    NStateInterest=0
    do ii=1,Nlist_K
        do iband=1,Nbands
            if (abs(el_energy(ii,iband)).le.Ethreshold) then
                NStateInterest=NStateInterest+1
            endif
        enddo
    enddo

    allocate(StateInterest(2,NStateInterest))

    NStateInterest=0
    do ii=1,Nlist_K
        do iband=1,Nbands
            if (abs(el_energy(ii,iband)).le.Ethreshold) then
                NStateInterest=NStateInterest+1
                StateInterest(1,NStateInterest)=iband
                StateInterest(2,NStateInterest)=ii
            endif
        enddo
    enddo

    Nstates=ceiling(float(NStateInterest)/nproc)
!!------------------------------------------------------------------------------
end subroutine ElectronInterest
!!------------------------------------------------------------------------------

!!------------------------------------------------------------------------------
SUBROUTINE prepare_lattice
!!------------  ------------------------------------------------------------------
    !
    USE global_variables  
    USE global_constants  
    USE symm_base,        ONLY : irt, s, nsym, ftau, sname, invs, s_axis_to_cart,&
                                  sr, nrot, copy_sym, set_sym_bl, find_sym, inverse_s,remove_sym, allfrac
    USE ions_base,        ONLY : nat, nsp, tau, ityp
    USE cell_base,           only: at, celldm
    USE noncollin_module, ONLY : m_loc
    USE fft_base,         ONLY : dfftp
    USE mp_world,         only : mpime
    USE constraints_module, ONLY : cross
    implicit none
    integer :: ii,jj
    integer :: i,j,k
    logical :: timereversal=.true. ! by default, time reversal symmetry is required to be included.
    real(kind=8) :: invrlattvec(3,3), aa(3,3)
    !
    alat=celldm(1)
    lattvec=at*celldm(1)
    lattvec=lattvec*bohr2nm
    do i=1,3
        j=mod(i,3)+1
        k=mod(j,3)+1
        rlattvec(:,i) = cross(lattvec(:,j),lattvec(:,k),)
    enddo
    Vol=abs(DDOT(3, (lattvec(:,1),rlattvec(:,1))))
    rlattvec=2.d0*pi/Vol*rlattvec    ! in unit of 1/nm
    !
    CALL set_sym_bl ( ) ! This should define the s matrix
    CALL find_sym ( nat, tau, ityp, .false., m_loc )
    !
    Nsymm=nsym
    !
    allocate(Orthcar(3,3,Nsymm))
    allocate(symm_rev_idx(Nsymm))
    Orthcar=sr(:,:,1:nsym)
    !
    !! transport symmetry operation to crystal lattice vector
    allocate(Orth(3,3,Nsymm))
    allocate(Orth_rev(3,3,Nsymm))
    invrlattvec=rlattvec
    call invmatrix(invrlattvec)
    do ii=1,Nsymm
        Orth(:,:,ii)=matmul(invrlattvec,matmul(Orthcar(:,:,ii),rlattvec))
    enddo
    !
    do ii=1,Nsymm
        do jj=1,Nsymm
            if (maxval(abs(matmul(Orth(:,:,ii),Orth(:,:,jj))-aa)).lt.1.d-5) then
                symm_rev_idx(ii) = jj
                Orth_rev(:,:,ii) = Orth(:,:,jj)
            end if
        end do
    end do
    !
END SUBROUTINE prepare_lattice

!!------------------------------------------------------------------------------
subroutine get_sym()
!!------------------------------------------------------------------------------
    USE global_variables  
    USE symm_base,        ONLY : irt, s, nsym, ftau, sname, invs, s_axis_to_cart,&
                                  sr, nrot, copy_sym, set_sym_bl, find_sym, inverse_s,remove_sym, allfrac
    USE ions_base,        ONLY : nat, nsp, tau, ityp
    USE noncollin_module, ONLY : m_loc
    USE fft_base,         ONLY : dfftp
    use lattice_variables,only : rlattvec
    USE mp_world,         only : mpime
    implicit none
    integer :: ii,jj
    logical :: timereversal=.true. ! by default, time reversal symmetry is required to be included.
    real(kind=8) :: invrlattvec(3,3),aa(3,3)
    !
    CALL set_sym_bl ( ) ! This should define the s matrix
    CALL find_sym ( nat, tau, ityp, .false., m_loc )
    !
    Nsymm=nsym
    !
    allocate(Orthcar(3,3,Nsymm))
    allocate(symm_rev_idx(Nsymm))
    Orthcar=sr(:,:,1:nsym)
    !
    !! transport symmetry operation to crystal lattice vector
    allocate(Orth(3,3,Nsymm))
    allocate(Orth_rev(3,3,Nsymm))
    invrlattvec=rlattvec
    call invmatrix(invrlattvec)
    do ii=1,Nsymm
        Orth(:,:,ii)=matmul(invrlattvec,matmul(Orthcar(:,:,ii),rlattvec))
    enddo
    !
    do ii=1,Nsymm
        do jj=1,Nsymm
            if (maxval(abs(matmul(Orth(:,:,ii),Orth(:,:,jj))-aa)).lt.1.d-5) then
                symm_rev_idx(ii) = jj
                Orth_rev(:,:,ii) = Orth(:,:,jj)
            end if
        end do
    end do
    !
    if (mpime.eq.0) write(*,'("Info: Number of symmetry",I6)') Nsymm
    !
end subroutine get_sym

subroutine invmatrix(a)
    implicit none
    real(kind=8) :: a(3,3)

    integer :: ipiv(3), info
    real(kind=8) :: work(3)

    call dgetrf( 3, 3, a, 3, ipiv, info )
    call dgetri( 3, a, 3, ipiv, work, 3, info )

    return
end subroutine invmatrix





