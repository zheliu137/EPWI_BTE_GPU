MODULE global_constants
    real(kind=8),parameter :: &
            sr2 = 1.414213562373d0,&
            sr3 = 1.732050807569d0,&
            pi=3.141592653589793238d0,&
            kb=8.6173324d-5,& ! ev/K !kb=1.380648813d-23 J/K
            hbar=1.05457172647d-22,& ! J/THz
            hbar_js=1.05457172647d-34,& ! J*s
            echarge=1.60217657d-19,& ! coulombs
            bohr2nm=0.052917721092,&
            Ry2ev=1.360569253d1,&
            THz2ev=4.13566750456d-3,&
            radps2ev=4.13566750456d-3/2.d0/pi,&
            cm2ev=1.2398d-4, &
            ryd2ms1=1.0938457d6
    INTEGER,PARAMETER :: LI = 8 ! long int 
END MODULE global_constants

MODULE Scattering_vars
    USE global_constants, ONLY: LI
    IMPLICIT NONE
    !
    integer :: Nsymm
    integer,ALLOCATABLE :: symm_rev_idx(:)
    real(kind=8),allocatable :: Orth(:,:,:),Orthcar(:,:,:)
    real(kind=8),allocatable :: Orth_rev(:,:,:)
    !
    real(kind=8),allocatable :: el_energy(:,:), el_velocity(:,:,:) !energy(nptk_K,Nbands),el_velocity(nptk_K,Nbands,3)
    real(kind=8),allocatable :: ph_energy(:,:), ph_velocity(:,:,:) !ph_omega(nptk_Q,Nmodes),ph_velocity(nptk_Q,Nmodes,3)
    integer,allocatable :: StateInterest(:,:) !StateInterest(2,NPTK_K*Nbands)==>(2,NStateInterest)
    integer :: NStateInterest, Nstates !! NstatesAdd,  Nstates is per pool
    integer,allocatable :: PhononInterest(:)
    integer :: NPhononInterest
    integer :: NList_K,NList_Q
    integer,allocatable :: List_K(:),Eqindex_K(:,:),List_Q(:),Eqindex_Q(:,:)
    real(kind=8) :: lattvec(3,3),rlattvec(3,3),alat,Vol
    real(kind=8) :: DosFermi, V2Fermi(3)
    real(kind=8) :: spin_degen
    integer(kind=LI), allocatable :: naccum(:)
    integer(kind=4),  allocatable :: indscatt(:)
    real(kind=8),     allocatable :: Gamma(:)
    real(kind=8),     allocatable :: rate_scatt(:),rate_scatt_mrta(:)
END MODULE Scattering_vars

MODULE legacy_variables
   !
   USE kinds,        ONLY: DP
   USE pwcom,        ONLY: nbnd, nks, nkstot, isk, et, xk, ef, nelec
   USE cell_base,    ONLY: at, bg, omega, alat
   USE start_k,      ONLY: nk1, nk2, nk3
   USE ions_base,    ONLY: nat, amass, ityp, tau, ntyp => nsp
   USE phcom,        ONLY: nq1, nq2, nq3, nmodes
   USE epwcom,       ONLY: nq1_ifc, nq2_ifc, nq3_ifc
   USE epwcom,       ONLY: nbndsub, lrepmatf, fsthick, epwread, longrange, &
                           epwwrite, ngaussw, degaussw, lpolar, lifc, &
                           nbndskip, parallel_k, parallel_q, etf_mem, &
                           elecselfen, phonselfen, nest_fn, a2f, &
                           vme, eig_read, ephwrite, nkf1, nkf2, nkf3, &
                           efermi_read, fermi_energy, specfun_el, band_plot, &
                           nqf1, nqf2, nqf3, mp_mesh_k, restart, eps_acustic, &
                           filqf, filkf
   USE noncollin_module, ONLY: noncolin
   USE constants_epw,    ONLY: ryd2ev, ryd2mev, one, two, czero, twopi, ci, zero
   USE io_files,         ONLY: prefix, diropn
   USE io_global,        ONLY: stdout, ionode
   USE io_epw,           ONLY: lambda_phself, linewidth_phself, iunepmatwe, &
                               iunepmatwp, crystal
   USE elph2,            ONLY: nrr_k, nrr_q, cu, cuq, lwin, lwinq, irvec, ndegen_k, &
                               ndegen_q, wslen, chw, chw_ks, cvmew, cdmew, rdw, &
                               epmatwp, epmatq, wf, etf, etf_k, etf_ks, xqf, xkf, &
                               wkf, dynq, nqtotf, nkqf, epf17, nkf, nqf, et_ks, &
                               ibndmin, ibndmax, lambda_all, dmec, dmef, vmef, &
                               sigmai_all, sigmai_mode, gamma_all, epsi, zstar, &
                               efnew, ifc, sigmar_all, zi_all, nkqtotf
  USE clib_wrappers,     ONLY: f_mkdir_safe
  !
  USE mp,                ONLY: mp_barrier, mp_bcast, mp_sum
  USE io_global,         ONLY: ionode_id
  USE mp_global,         ONLY: inter_pool_comm, intra_pool_comm, root_pool
  USE mp_world,          ONLY: mpime
  !
  IMPLICIT NONE
  !
  INTEGER :: nrws
  !! Number of real-space Wigner-Seitz
  INTEGER, PARAMETER :: nrwsx = 200
  !! Maximum number of real-space Wigner-Seitz
  INTEGER :: valueRSS(2)
  !! Return virtual and resisdent memory from system
  REAL(kind=DP) :: xxq(3)
  !! Current q-point
  REAL(kind=DP) :: xxk(3)
  !! Current k-point on the fine grid
  REAL(kind=DP) :: xkk(3)
  !! Current k-point on the fine grid
  REAL(kind=DP) :: xkq(3)
  !! Current k+q point on the fine grid
  REAL(kind=DP) :: rws(0:3, nrwsx)
  !! Real-space wigner-Seitz vectors
  REAL(kind=DP) :: atws(3, 3)
  !! Maximum vector: at*nq
  REAL(kind=DP), PARAMETER :: eps = 0.01/ryd2mev
  !! Tolerence
  REAL(kind=DP), ALLOCATABLE :: w2(:)
  !! Interpolated phonon frequency
  REAL(kind=DP), ALLOCATABLE :: irvec_r(:, :)
  !! Wigner-Size supercell vectors, store in real instead of integer
  REAL(kind=DP), ALLOCATABLE :: rdotk(:)
  !! $r\cdot k$
  !
  COMPLEX(kind=DP), ALLOCATABLE :: epmatwe(:, :, :, :, :)
  !! e-p matrix  in wannier basis - electrons
  COMPLEX(kind=DP), ALLOCATABLE :: epmatwef(:, :, :, :)
  !! e-p matrix  in el wannier - fine Bloch phonon grid
  COMPLEX(kind=DP), ALLOCATABLE :: epmatf(:, :, :)
  !! e-p matrix  in smooth Bloch basis, fine mesh
  COMPLEX(kind=DP), ALLOCATABLE :: cufkk(:, :)
  !! Rotation matrix, fine mesh, points k
  COMPLEX(kind=DP), ALLOCATABLE :: cufkq(:, :)
  !! the same, for points k+q
  COMPLEX(kind=DP), ALLOCATABLE :: uf(:, :)
  !! Rotation matrix for phonons

END MODULE legacy_variables

MODULE global_variables
  !
  USE legacy_variables
  USE Scattering_vars
  !
END MODULE global_variables



MODULE newinputs
    IMPLICIT NONE
    character(len=20) :: prefix
    character(len=20) :: type_key

    character(len=20) :: weighttype

    integer :: Nmodes
    integer :: Nbands             ! Number of bands of interest, which should be consistent with the input of el-ph coupling elements
    integer :: Nbands_irrelevant  ! number of the first few irrelavant bands in the "bands.dat" file
    integer :: NKgrid(3),NPTK_K
    integer :: NQgrid(3),NPTK_Q
    real(kind=8) :: Nbands_filled ! number (can be fractional) of filled bands of interest, which is not used when metal_key=.false.
    integer :: iweight

    real(kind=8) :: Te
    real(kind=8) :: scalebroad
    real(kind=8) :: Ethreshold    ! above and below the fermi level in metals, and above the band minimum in n-type semiconductors
    real(kind=8) :: ph_cut        ! phonon acoustic cutoff, in the unit of eV
    real(kind=8) :: delta_mult    ! multiplied with broadening of gaussian as the criteria of delta function

    real(kind=8) :: tolerance
    integer :: maxiter
    logical :: convergence

    real(kind=8) :: ChemPot, EFermi, delta_shift_EFermi
    integer :: ismear_ecp ! smearing method for energy conservation principles
    real(kind=8) :: degauss ! constant broadening in unit of eV
    integer :: batchsize ! GPU batch size 

contains

    subroutine read_newinputs()
        USE epwcom,     ONLY : nkf1, nkf2, nkf3, nqf1, nqf2, nqf3, filqf, lpolar
        USE ions_base,  ONLY : nat
        USE mp_world,   ONLY : mpime
        USE noncollin_module, ONLY : noncolin
        USE elph_variables,   ONLY : spin_degen
        implicit none
        integer :: ii
        logical, external :: imatches

        namelist /systems/ prefix,type_key,Nbands,Nbands_irrelevant,Nbands_filled, &
                           Te,scalebroad,Ethreshold, ismear_ecp, ChemPot, &
                           EFermi, delta_shift_EFermi, degauss, batchsize, ph_cut, & 
                           delta_mult, convergence,tolerance,maxiter

        open(1,file="./inputs/CONTROL",status="old")
        type_key='' 
        Nbands=-1
        Nbands_irrelevant=-1
        degauss = 0.01
        Ethreshold=0.0
        scalebroad=1.0
        ph_cut = 1e-4
        Te=300.
        EFermi=-1.0D20
        delta_shift_EFermi=0.0
        batchsize=3000
        ismear_ecp=0
        delta_mult=2.0
        convergence=.true.
        tolerance=1.d-3
        maxiter=50
        read(1,nml=systems)
        close(1)

        NKgrid(1)=nkf1
        NKgrid(2)=nkf2
        NKgrid(3)=nkf3

        NQgrid(1)=nqf1
        NQgrid(2)=nqf2
        NQgrid(3)=nqf3

        NPTK_K=NKgrid(1)*NKgrid(2)*NKgrid(3)
        NPTK_Q=NQgrid(1)*NQgrid(2)*NQgrid(3)

        if (noncolin) then
            spin_degen=1.0
        else
            spin_degen=2.0
        endif

    end subroutine read_newinputs

END MODULE newinputs


