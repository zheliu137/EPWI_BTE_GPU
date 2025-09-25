# Makefile for EPW
RELPATH=../../../..

include $(RELPATH)/make.inc
include make.libs

#
# use recursive definitions in Makefile to modify the PATH for modules
#

IFLAGS   = -I$(RELPATH)/include  -I$(RELPATH)/UtilXlib/
MODFLAGS = -I$(RELPATH)/iotk/src -I$(RELPATH)/UtilXlib/ -I$(RELPATH)/Modules -I$(RELPATH)/KS_Solvers/CG -I$(RELPATH)/KS_Solvers/Davidson \
           -I$(RELPATH)/LAXlib -I$(RELPATH)/FFTXlib  -I$(RELPATH)/PW/src -I$(RELPATH)/LR_Modules -I$(RELPATH)/PHonon/PH \
	   -I$(RELPATH)/PP/src/ -I. -I$(RELPATH)/wannier90-2.1.0/src/obj/

#toggle debug in Fortran code
DFLAGS+=-D__GPU -D__DEBUG
# DFLAGS+=-D__GPU

#
# The above is in alphabetical order except when order matter during compilation
#

EPWOBJS = \
transportcom.o \
io_epw.o\
elph2.o \
a2f.o \
allocate_epwq.o \
bcast_epw_input.o \
broyden.o \
close_epw.o \
constants_epw.o \
create_mesh.o \
createkmap.o \
deallocate_epw.o \
deallocate_eliashberg.o \
distribution.o \
dmebloch2wan.o \
dmewan2bloch.o \
dvanqq2.o \
dvqpsi_us3.o \
dvqpsi_us_only3.o \
dynbloch2wan.o \
dynwan2bloch.o \
eliashberg.o \
eliashbergcom.o \
eliashberg_aniso_cont_raxis.o \
eliashberg_aniso_iaxis.o \
eliashberg_iso_cont_raxis.o \
eliashberg_iso_iaxis.o \
eliashberg_iso_raxis.o \
eliashberg_pp.o \
eliashberg_readfiles.o \
eliashberg_setup.o \
eliashberg_write.o \
elphel2_shuffle.o \
elphon_shuffle.o \
elphon_shuffle_wrap.o \
ephbloch2wane.o \
ephbloch2wanp.o \
ephwan2bloch.o \
ephwan2bloch_mem.o \
ephwan2blochp.o \
ephwan2blochp_mem.o \
ephwann_shuffle.o \
ephwann_shuffle_mem.o \
epwcom.o \
epw_init.o \
epw_readin.o \
epw_setup.o \
epw_summary.o \
fermiwindow.o \
gen_freqgrid.o \
gmap_sym.o \
hambloch2wan.o \
hamwan2bloch.o \
io_dyn_mat2.o \
kernels_aniso_iaxis.o \
kernels_iso_iaxis.o \
kernels_raxis.o \
kfold.o \
kpointdivision.o \
ktokpmq.o \
loadkmesh.o \
loadqmesh.o \
loadumat.o \
nesting_fn.o \
openfilepw.o \
rgd_blk_epw_fine_mem.o \
pade.o \
plot_band.o \
poolgather.o \
print_clock_epw.o \
print_gkk.o \
readdvscf.o \
readgmap.o \
readmat_shuffle2.o \
readwfc.o \
refold.o \
rigid_epw.o \
rotate_eigenm.o \
rotate_epmat.o \
selfen_elec.o \
selfen_phon.o \
selfen_pl.o \
set_ndnmbr.o \
setphases_wrap.o \
sgama2.o \
sort.o \
spectral_cumulant.o \
spectral_func.o \
spectral_func_ph.o \
spectral_func_pl.o \
star_q2.o \
stop_epw.o \
vmebloch2wan.o \
vmewan2bloch.o \
wannier.o \
wannierize.o \
pw2wan90epw.o \
wigner_seitz2.o \
wigner_seitz.o \
write_ephmat.o \
io_scattering.o \
system_mem_usage.o \
EPWI_run.o \
EPWI_globalvars.o \
EPWI_readins.o

# setphases.o \ : SP: We keep it in case its usefull in the future but now depreciated.

#default : epw


PHOBJS = $(RELPATH)/PHonon/PH/libph.a
PWOBJS = $(RELPATH)/PW/src/libpw.a
W90LIB = $(RELPATH)/wannier90-2.1.0/libwannier.a
LRMODS = $(RELPATH)/LR_Modules/liblrmod.a
PWOBJS = $(RELPATH)/PW/src/libpw.a
QEMODS = $(RELPATH)/Modules/libqemod.a $(RELPATH)/KS_Solvers/CG/libcg.a  $(RELPATH)/KS_Solvers/Davidson/libdavid.a $(RELPATH)/LAXlib/libqela.a \
         $(RELPATH)/FFTXlib/libqefft.a
LIBOBJS =$(RELPATH)/UtilXlib/libutil.a $(RELPATH)/clib/clib.a $(RELPATH)/iotk/src/libiotk.a

TLDEPS= bindir mods libs pw-lib pw ph

#all :   pw ph wannier wcorr pp ld1 upf libepw.a epw.x
all :   pw ph wannier pp ld1 upf libepw.a epw.x

## GPU
#uncomment FLAGS to toggle debug in cuda(C/C++) code
# GPU_FLAGS+=-D DEBUG
GPU_FLAGS+=-g -O2
# GPU_FLAGS+=-g -O1
NVCC_FLAGS+=--default-stream per-thread -gencode=arch=compute_70,code=sm_70
GPU_LIBS+=-L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver -static-libstdc++ -cxxlib
GPU_OBJS+=\
cuda_kernel.o \
cuda_snippets.o\
cuda_elphwann_wannier.o\
cuda_band_wannier.o\
cuda_wedge.o\
cuda_Nprocesses.o \
cuda_debug.o \
cuda_utils.o \
cuda_check_valid.o

cuda_kernel.o: ./cuf/cuda_kernel.cu
	nvcc $(GPU_FLAGS) $(NVCC_FLAGS) -c -o $@ $<
cuda_snippets.o: ./cuf/cuda_snippets.cu
	nvcc $(GPU_FLAGS) $(NVCC_FLAGS) -c -o $@ $<
cuda_elphwann_wannier.o: ./cuf/cuda_elphwann_wannier.cu
	nvcc $(GPU_FLAGS) $(NVCC_FLAGS) -c -o $@ $<
cuda_band_wannier.o: ./cuf/cuda_band_wannier.cu
	nvcc $(GPU_FLAGS) $(NVCC_FLAGS) -c -o $@ $<
cuda_wedge.o: ./cuf/cuda_wedge.cu
	nvcc $(GPU_FLAGS) $(NVCC_FLAGS) -c -o $@ $<
cuda_Nprocesses.o: ./cuf/cuda_Nprocesses.cu
	nvcc $(GPU_FLAGS) $(NVCC_FLAGS) -c -o $@ $<
cuda_debug.o: ./cuf/cuda_debugs.cu
	nvcc $(GPU_FLAGS) $(NVCC_FLAGS) -c -o $@ $<
cuda_utils.o: ./cuf/cuda_utils.cu
	nvcc $(GPU_FLAGS) $(NVCC_FLAGS) -c -o $@ $<
cuda_check_valid.o: ./cuf/cuda_check_valid.cu
	nvcc $(GPU_FLAGS) $(NVCC_FLAGS) -c -o $@ $<

libepw.a : $(GPU_OBJS) $(EPWOBJS)
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@

epw.x : epw.o libepw.a $(PHOBJS) $(LRMODS) $(PWOBJS) $(W90LIB) $(QEMODS) $(LIBOBJS)
	$(LD) $(LDFLAGS) -o $@ \
	epw.o  libepw.a	$(PHOBJS) $(LRMODS) $(W90LIB) $(PWOBJS) $(QEMODS) $(LIBOBJS) $(LIBS) $(GPU_LIBS)
	# - ( cd ../bin ; ln -fs ../src/epw.x . )

# SP: Here to correct bugs in wannier 2.0.1
wcorr :
	sed -i "s/allocate(fermi_energy_list(nfermi),stat=ierr)/if (.not. allocated(fermi_energy_list) ) allocate(fermi_energy_list(nfermi))/g" $(RELPATH)/wannier90-2.0.1/src/parameters.F90 ; sed -i "s/allocate(kubo_freq_list(kubo_nfreq),stat=ierr)/if (.not. allocated(kubo_freq_list) ) allocate(kubo_freq_list(kubo_nfreq)) /g" $(RELPATH)/wannier90-2.0.1/src/parameters.F90 ; cp wannier_lib.f90 wannier_lib.F90 ; cp wannier_lib.F90 $(RELPATH)/wannier90-2.0.1/src/

pw :
	cd $(RELPATH)/ ; make pw

ph :
	cd $(RELPATH)/ ; make ph

wannier :
	cd $(RELPATH)/ ; make w90 ; cd wannier90-2.1.0/ ; make lib

pp :
	cd $(RELPATH)/ ; make pp

ld1 :
	cd $(RELPATH)/ ; make ld1

upf :
	cd $(RELPATH)/ ; make upf

clean :
	- /bin/rm -f  *.o *~ *.d *.mod *.i libepw.a liblr.a

include make.depend
