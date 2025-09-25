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
