#include "gpuec256.h"

#include "device_arr.h"
#include "constants.cu"
// typedef unsigned long long UINT64; //????64��??????
// typedef long long INT64;
// typedef unsigned int UINT32;
// // ???????
// typedef struct Affine_point{
// 	UINT64 x[8];
// 	UINT64 y[8];
// }Apoint;

// // ???????
// typedef struct Jacobi_point{
// 	UINT64 x[8];
// 	UINT64 y[8];
// 	UINT64 z[8];
// }Jpoint;


// ??tesla C2050??????????????��??????????????
#define PARAL 64
#define BLOCKNUM (14*8)
#define BLOCKSIZE 32
#define THREADNUM (BLOCKNUM*BLOCKSIZE)


// ????__global__?????????????16???????
#define d_BIN_WINDOW_16 16 //16?????
#define d_ROWS_16 32
#define d_COLS_16 (1L<<d_BIN_WINDOW_16)


#define HANDLE_ERROR( err ) { if (err != cudaSuccess) { \
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), __FILE__, __LINE__ );\
	  exit( EXIT_FAILURE ); }  \
}

//?��?512??????????????????????256????��
//??��?????????????????????????
//h_ONE??host????????1
//dc_ONE??gpu??????????1
//h_mon_ONE??dc_mon_ONE?????????host??gpu???????????????????1
//h_p??dc_p??512????????
//Pa0??Pa7???????????????????

// const UINT64 h_ONE[8]={0x0000000000000001L,0x0000000000000000L,0x0000000000000000L,0x0000000000000000L,0x0000000000000000L,0x0000000000000000L,0x0000000000000000L,0x0000000000000000L};
// const UINT64 h_mon_ONE[8]={0x0000000000000001L,0x0000000100000000L,0x0000000000000000L,0x0000000100000000L,0x0000000000000000L,0x0000000000000001L,0x0000000100000000L,0x0000000000000000L};
// const UINT64 h_p[8]={0xFFFFFFFEFFFFFC2FL,0xFFFFFFFFFFFFFFFFL,0xFFFFFFFFFFFFFFFFL,0xFFFFFFFFFFFFFFFFL,0x0L,0x0L,0x0L,0x0L};

// __constant__ UINT64 dc_ONE[8]={0x0000000000000001L,0x0000000000000000L,0x0000000000000000L,0x0000000000000000L,0x0000000000000000L,0x0000000000000000L,0x0000000000000000L,0x0000000000000000L};
// __constant__ UINT64 dc_mon_ONE[8]={0x0000000000000001L,0x0000000100000000L,0x0000000000000000L,0x0000000100000000L,0x0000000000000000L,0x0000000000000001L,0x0000000100000000L,0x0000000000000000L};
// __constant__ UINT64 dc_p[8]={0xFFFFFFFEFFFFFC2FL,0xFFFFFFFFFFFFFFFFL,0xFFFFFFFFFFFFFFFFL,0xFFFFFFFFFFFFFFFFL,0x0L,0x0L,0x0L,0x0L};


// const UINT64 h_ONE[4]={0x0000000000000001L,0x0000000000000000L,0x0000000000000000L,0x0000000000000000L};
// const UINT64 h_mon_ONE[4]={0x1000003d1L,0x0L,0x0L,0x0L};
// const UINT64 h_p[4]={0xFFFFFFFEFFFFFC2FL,0xFFFFFFFFFFFFFFFFL,0xFFFFFFFFFFFFFFFFL,0xFFFFFFFFFFFFFFFFL};

// __constant__ UINT64 dc_ONE[4]={0x0000000000000001L,0x0000000000000000L,0x0000000000000000L,0x0000000000000000L};
// __constant__ UINT64 dc_mon_ONE[4]={0x1000003d1L,0x0L,0x0L,0x0L};
// __constant__ UINT64 dc_p[4]={0xFFFFFFFEFFFFFC2FL,0xFFFFFFFFFFFFFFFFL,0xFFFFFFFFFFFFFFFFL,0xFFFFFFFFFFFFFFFFL};

//__constant__ UINT64 dc_mon_inv_two[4]={0x0L,0x0L,0x0L,0x8000000000000000L};
//const UINT64 h_mon_inv_two[4]={0x0L,0x0L,0x0L,0x8000000000000000L};

#define Pa0 0x3c208c16d87cfd47ULL //-1
#define Pa1 0x97816a916871ca8dULL 
#define Pa2 0xb85045b68181585dULL //-1
#define Pa3 0x30644e72e131a029ULL 
// #define Pa4 0x0 //-1
// #define Pa5 0x0 //-2
// #define Pa6 0x0 
// #define Pa7 0x0 //-1
#define Pn0 0x43e1f593f0000001ULL //?????????pn0????dc_N
#define Pn1 0x2833e84879b97091ULL
#define Pn2 0xb85045b68181585dULL
#define Pn3 0x30644e72e131a029ULL


/////////////////GPU???????????????////////////////////////////////////////////////////

// #define dh_mybig_copy(a,b) {(a)[0]=(b)[0];(a)[1]=(b)[1];(a)[2]=(b)[2];(a)[3]=(b)[3];(a)[4]=(b)[4];(a)[5]=(b)[5];(a)[6]=(b)[6];(a)[7]=(b)[7];}
#define dh_mybig_copy(a,b) {(a)[0]=(b)[0];(a)[1]=(b)[1];(a)[2]=(b)[2];(a)[3]=(b)[3];}

//?????????????????
__device__ __host__ void dh_mybig_modadd_64(const UINT64 *x, const UINT64 *y, UINT64 *z)//????z=x+y, x=x+y, ???????y=x+y??
{
	int i;
	UINT64 f,g;
		
	#ifdef __CUDA_ARCH__
		const UINT64 *Pa=dc_p;
	#else
		const UINT64 *Pa=h_p;
	#endif	
	
	z[0] = x[0] + y[0]; f = z[0] < y[0];
	z[1] = x[1] + f; g = z[1] < f; z[1] += y[1]; g += z[1] < y[1];
	z[2] = x[2] + g; f = z[2] < g; z[2] += y[2]; f += z[2] < y[2];
	z[3] = x[3] + f; g = z[3] < f; z[3] += y[3]; g += z[3] < y[3];
	
	// z[4] = x[4] + g; f = z[4] < g; z[4] += y[4]; f += z[4] < y[4];
	// z[5] = x[5] + f; g = z[5] < f; z[5] += y[5]; g += z[5] < y[5];
	// z[6] = x[6] + g; f = z[6] < g; z[6] += y[6]; f += z[6] < y[6];
	// z[7] = x[7] + f; g = z[7] < f; z[7] += y[7]; g += z[7] < y[7];
		
	if(g==0)
	{
		for(i=3;i>=0;i--)
		{
			if(z[i]!=Pa[i])
			{
				g=(z[i]>Pa[i]);
				break;
			}
			else if(i==0)//????,??t=P, ??????????0????
			{
				g=1;
			}
		}
	}
	
	if(g)//x+y??????????
	{
		f = z[0] < Pa0; z[0] -= Pa0;
		g = z[1] < f; z[1] -= f; g += z[1] < Pa1; z[1] -= Pa1;
		f = z[2] < g; z[2] -= g; f += z[2] < Pa2; z[2] -= Pa2;
		z[3] -= f; z[3] -= Pa3;		                                                  
		// g = z[3] < f; z[3] -= f; g += z[3] < Pa3; z[3] -= Pa3;		
		// f = z[4] < g; z[4] -= g; f += z[4] < Pa4; z[4] -= Pa4;		                                                  
		// g = z[5] < f; z[5] -= f; g += z[5] < Pa5; z[5] -= Pa5;
		// f = z[6] < g; z[6] -= g; f += z[6] < Pa6; z[6] -= Pa6;				
		// z[7] -= f; z[7] -= Pa7;
	}
}

__device__ __host__ void dh_mybig_modadd_64_modN(const UINT64 *x, const UINT64 *y, UINT64 *z)//????z=x+y, x=x+y, ???????y=x+y??
{
	int i;
	UINT64 f,g;
		
	#ifdef __CUDA_ARCH__
		const UINT64 *Pa=dc_N;
	#else
		const UINT64 *Pa=h_N;
	#endif	
	
	z[0] = x[0] + y[0]; f = z[0] < y[0];
	z[1] = x[1] + f; g = z[1] < f; z[1] += y[1]; g += z[1] < y[1];
	z[2] = x[2] + g; f = z[2] < g; z[2] += y[2]; f += z[2] < y[2];
	z[3] = x[3] + f; g = z[3] < f; z[3] += y[3]; g += z[3] < y[3];
	
	// z[4] = x[4] + g; f = z[4] < g; z[4] += y[4]; f += z[4] < y[4];
	// z[5] = x[5] + f; g = z[5] < f; z[5] += y[5]; g += z[5] < y[5];
	// z[6] = x[6] + g; f = z[6] < g; z[6] += y[6]; f += z[6] < y[6];
	// z[7] = x[7] + f; g = z[7] < f; z[7] += y[7]; g += z[7] < y[7];
		
	if(g==0)
	{
		for(i=3;i>=0;i--)
		{
			if(z[i]!=Pa[i])
			{
				g=(z[i]>Pa[i]);
				break;
			}
			else if(i==0)//????,??t=P, ??????????0????
			{
				g=1;
			}
		}
	}
	
	if(g)//x+y??????????
	{
		f = z[0] < Pa[0]; z[0] -= Pa[0];
		g = z[1] < f; z[1] -= f; g += z[1] < Pa[1]; z[1] -= Pa[1];
		f = z[2] < g; z[2] -= g; f += z[2] < Pa[2]; z[2] -= Pa[2];
		z[3] -= f; z[3] -= Pa[3];		                                                  
		// g = z[3] < f; z[3] -= f; g += z[3] < Pa3; z[3] -= Pa3;		
		// f = z[4] < g; z[4] -= g; f += z[4] < Pa4; z[4] -= Pa4;		                                                  
		// g = z[5] < f; z[5] -= f; g += z[5] < Pa5; z[5] -= Pa5;
		// f = z[6] < g; z[6] -= g; f += z[6] < Pa6; z[6] -= Pa6;				
		// z[7] -= f; z[7] -= Pa7;
	}
}

//??????????????????
__device__ __host__ void dh_mybig_modsub_64(const UINT64 *x, const UINT64 *y, UINT64 *z)//????z=x-y, x=x-y, ???????y=x-y??
{
	UINT64 f,g;
	//UINT64 z0,z1,z2,z3,z4,z5,z6,z7;
	f=(x[0]<y[0]); z[0]=x[0]-y[0];
	g=(x[1]<f); z[1]=x[1]-f; g+=(z[1]<y[1]); z[1]-=y[1];
	f=(x[2]<g); z[2]=x[2]-g; f+=(z[2]<y[2]); z[2]-=y[2];
	g=(x[3]<f); z[3]=x[3]-f; g+=(z[3]<y[3]); z[3]-=y[3];
	
	// f=(x[4]<g); z[4]=x[4]-g; f+=(z[4]<y[4]); z[4]-=y[4];
	// g=(x[5]<f); z[5]=x[5]-f; g+=(z[5]<y[5]); z[5]-=y[5];
	// f=(x[6]<g); z[6]=x[6]-g; f+=(z[6]<y[6]); z[6]-=y[6];
	// g=(x[7]<f); z[7]=x[7]-f; g+=(z[7]<y[7]); z[7]-=y[7];
	//??��?g;

	if(g)//??��??��????????????
	{
		z[0]+=Pa0; f=(z[0]<Pa0);
		z[1]+=f; f=(z[1]<f); z[1]+=Pa1; f+=(z[1]<Pa1);
		z[2]+=f; f=(z[2]<f); z[2]+=Pa2; f+=(z[2]<Pa2);
		z[3]+=f; z[3]+=Pa3;		
		// z[3]+=f; f=(z[3]<f); z[3]+=Pa3; f+=(z[3]<Pa3);
		// z[4]+=f; f=(z[4]<f); z[4]+=Pa4; f+=(z[4]<Pa4);
		// z[5]+=f; f=(z[5]<f); z[5]+=Pa5; f+=(z[5]<Pa5);
		// z[6]+=f; f=(z[6]<f); z[6]+=Pa6; f+=(z[6]<Pa6);
		// z[7]+=f; z[7]+=Pa7;		
	}	
}

//x-y mod N
__device__ __host__ void dh_mybig_modsub_64_modN(const UINT64 *x, const UINT64 *y, UINT64 *z){

	#ifdef __CUDA_ARCH__	
		UINT64 *Pa=dc_N;	
		#define h_Hi64 __umul64hi
	#else
		const UINT64 *Pa=h_N;
	#endif	
	UINT64 f,g;
	//UINT64 z0,z1,z2,z3,z4,z5,z6,z7;
	f=(x[0]<y[0]); z[0]=x[0]-y[0];
	g=(x[1]<f); z[1]=x[1]-f; g+=(z[1]<y[1]); z[1]-=y[1];
	f=(x[2]<g); z[2]=x[2]-g; f+=(z[2]<y[2]); z[2]-=y[2];
	g=(x[3]<f); z[3]=x[3]-f; g+=(z[3]<y[3]); z[3]-=y[3];
	
	// f=(x[4]<g); z[4]=x[4]-g; f+=(z[4]<y[4]); z[4]-=y[4];
	// g=(x[5]<f); z[5]=x[5]-f; g+=(z[5]<y[5]); z[5]-=y[5];
	// f=(x[6]<g); z[6]=x[6]-g; f+=(z[6]<y[6]); z[6]-=y[6];
	// g=(x[7]<f); z[7]=x[7]-f; g+=(z[7]<y[7]); z[7]-=y[7];
	//??��?g;

	if(g)//??��??��????????????
	{
		z[0]+=Pa[0]; f=(z[0]<Pa[0]);
		z[1]+=f; f=(z[1]<f); z[1]+=Pa[1]; f+=(z[1]<Pa[1]);
		z[2]+=f; f=(z[2]<f); z[2]+=Pa[2]; f+=(z[2]<Pa[2]);
		z[3]+=f; z[3]+=Pa[3];		
		// z[3]+=f; f=(z[3]<f); z[3]+=Pa3; f+=(z[3]<Pa3);
		// z[4]+=f; f=(z[4]<f); z[4]+=Pa4; f+=(z[4]<Pa4);
		// z[5]+=f; f=(z[5]<f); z[5]+=Pa5; f+=(z[5]<Pa5);
		// z[6]+=f; f=(z[6]<f); z[6]+=Pa6; f+=(z[6]<Pa6);
		// z[7]+=f; z[7]+=Pa7;		
	}	
}

__device__ __host__ void dh_mybig_modsub_64_ui32_modN(const UINT64 *x, unsigned int y, UINT64 *z){

	#ifdef __CUDA_ARCH__	
		UINT64 *Pa=dc_N;	
		#define h_Hi64 __umul64hi
	#else
		const UINT64 *Pa=h_N;
	#endif	
	UINT64 f,g;
	//UINT64 z0,z1,z2,z3,z4,z5,z6,z7;
	f=(x[0]<y); z[0]=x[0]-y;
	g=(x[1]<f); z[1]=x[1]-f; 
	f=(x[2]<g); z[2]=x[2]-g; 
	g=(x[3]<f); z[3]=x[3]-f; 
	
	// f=(x[4]<g); z[4]=x[4]-g; f+=(z[4]<y[4]); z[4]-=y[4];
	// g=(x[5]<f); z[5]=x[5]-f; g+=(z[5]<y[5]); z[5]-=y[5];
	// f=(x[6]<g); z[6]=x[6]-g; f+=(z[6]<y[6]); z[6]-=y[6];
	// g=(x[7]<f); z[7]=x[7]-f; g+=(z[7]<y[7]); z[7]-=y[7];
	//??��?g;

	if(g)//??��??��????????????
	{
		z[0]+=Pa[0]; f=(z[0]<Pa[0]);
		z[1]+=f; f=(z[1]<f); z[1]+=Pa[1]; f+=(z[1]<Pa[1]);
		z[2]+=f; f=(z[2]<f); z[2]+=Pa[2]; f+=(z[2]<Pa[2]);
		z[3]+=f; z[3]+=Pa[3];		
		// z[3]+=f; f=(z[3]<f); z[3]+=Pa3; f+=(z[3]<Pa3);
		// z[4]+=f; f=(z[4]<f); z[4]+=Pa4; f+=(z[4]<Pa4);
		// z[5]+=f; f=(z[5]<f); z[5]+=Pa5; f+=(z[5]<Pa5);
		// z[6]+=f; f=(z[6]<f); z[6]+=Pa6; f+=(z[6]<Pa6);
		// z[7]+=f; z[7]+=Pa7;		
	}	
}

__device__ __host__ void dh_mybig_neg(UINT64 *y,UINT64 *z){
	// #ifdef __CUDA_ARCH__	
	// 	UINT64 *Pa=dc_N;	
	// 	#define h_Hi64 __umul64hi
	// #else
	// 	const UINT64 *Pa=h_N;
	// #endif	
	UINT64 f,g;
	//UINT64 z0,z1,z2,z3,z4,z5,z6,z7;
	f=(0<y[0]); z[0]=0-y[0];
	g=(0<f); z[1]=0-f; g+=(z[1]<y[1]); z[1]-=y[1];
	f=(0<g); z[2]=0-g; f+=(z[2]<y[2]); z[2]-=y[2];
	g=(0<f); z[3]=0-f; g+=(z[3]<y[3]); z[3]-=y[3];
	

	//??��?g;

	if(g)//??��??��????????????
	{
		z[0]+=Pa0; f=(z[0]<Pa0);
		z[1]+=f; f=(z[1]<f); z[1]+=Pa1; f+=(z[1]<Pa1);
		z[2]+=f; f=(z[2]<f); z[2]+=Pa2; f+=(z[2]<Pa2);
		z[3]+=f; z[3]+=Pa3;				
		// z[3]+=f; f=(z[3]<f); z[3]+=Pa3; f+=(z[3]<Pa3);
		// z[4]+=f; f=(z[4]<f); z[4]+=Pa4; f+=(z[4]<Pa4);
		// z[5]+=f; f=(z[5]<f); z[5]+=Pa5; f+=(z[5]<Pa5);
		// z[6]+=f; f=(z[6]<f); z[6]+=Pa6; f+=(z[6]<Pa6);
		// z[7]+=f; z[7]+=Pa7;		
	}	
}

__device__ __host__ void dh_mybig_neg_modN(UINT64 *y,UINT64 *z){
	#ifdef __CUDA_ARCH__	
		UINT64 *Pa=dc_N;	
		#define h_Hi64 __umul64hi
	#else
		const UINT64 *Pa=h_N;
	#endif	
	UINT64 f,g;
	//UINT64 z0,z1,z2,z3,z4,z5,z6,z7;
	f=(0<y[0]); z[0]=0-y[0];
	g=(0<f); z[1]=0-f; g+=(z[1]<y[1]); z[1]-=y[1];
	f=(0<g); z[2]=0-g; f+=(z[2]<y[2]); z[2]-=y[2];
	g=(0<f); z[3]=0-f; g+=(z[3]<y[3]); z[3]-=y[3];
	

	//??��?g;

	if(g)//??��??��????????????
	{
		z[0]+=Pa[0]; f=(z[0]<Pa[0]);
		z[1]+=f; f=(z[1]<f); z[1]+=Pa[1]; f+=(z[1]<Pa[1]);
		z[2]+=f; f=(z[2]<f); z[2]+=Pa[2]; f+=(z[2]<Pa[2]);
		z[3]+=f; z[3]+=Pa[3];		
		// z[3]+=f; f=(z[3]<f); z[3]+=Pa3; f+=(z[3]<Pa3);
		// z[4]+=f; f=(z[4]<f); z[4]+=Pa4; f+=(z[4]<Pa4);
		// z[5]+=f; f=(z[5]<f); z[5]+=Pa5; f+=(z[5]<Pa5);
		// z[6]+=f; f=(z[6]<f); z[6]+=Pa6; f+=(z[6]<Pa6);
		// z[7]+=f; z[7]+=Pa7;		
	}	
}

//????????????????64��?????��
static inline UINT64 h_Hi64(const UINT64 x, const UINT64 y)
{
   UINT64 z;   
   __asm__ __volatile__ (
	  "movq  %0,%%rax\n"   
	  "mulq %1\n"
	  "movq %%rdx,%2\n"
	  : 
	  : "m"(x),"m"(y),"m"(z)
	  : "rax","rdx","memory"
	  );      	
	return z;
}

//?????????????C=a*b
/////////////////////////////////////////
//Montgomery???
//C=A*B*2^-512 mod P
//????CIOS??
//////////////////////////////////////////
__device__ __host__  void dh_mybig_monmult_64(const UINT64 *Aa, const UINT64 *Ba, UINT64 *Ca)//pass, c=a*b
{
	UINT64 t[4+2]={0};//8??64????????,???512
	
	//minv*P[0] mod 2^wordlen = -1. ????��????????64?????P[0]=0xffffffffffffffff=-1??????minv=1, minv*P[0]=-1 mod 2^64=-1
	
	//?????????P?????????????????��???????????????????????????
	// UINT64 minv=1;//minv????????????P?????��P[0]????2^64=-1(???P[0]??64????????????32?????????????2^32=-1??????? ????????????????????????2^wordlen?????????
	UINT64 minv = 0x87d20782e4866389;
	UINT64 m;	
	UINT64 c,s,cin;
	int i,j;
	
	#ifdef __CUDA_ARCH__	
		UINT64 *Pa=dc_p;	
		#define h_Hi64 __umul64hi
	#else
		const UINT64 *Pa=h_p;
	#endif	
	
	for(i=0;i<4;i++)
	{
		c=0;
		m=Ba[i];
		
		s=m*Aa[0]; c+=t[0];	cin=(c<t[0]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[0])+cin; 	t[0]=s;
		s=m*Aa[1]; c+=t[1];	cin=(c<t[1]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[1])+cin; 	t[1]=s;
		s=m*Aa[2]; c+=t[2];	cin=(c<t[2]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[2])+cin; 	t[2]=s;
		s=m*Aa[3]; c+=t[3];	cin=(c<t[3]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[3])+cin; 	t[3]=s;
		// s=m*Aa[4]; c+=t[4];	cin=(c<t[4]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[4])+cin; 	t[4]=s;
		// s=m*Aa[5]; c+=t[5];	cin=(c<t[5]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[5])+cin; 	t[5]=s;
		// s=m*Aa[6]; c+=t[6];	cin=(c<t[6]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[6])+cin; 	t[6]=s;
		// s=m*Aa[7]; c+=t[7];	cin=(c<t[7]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[7])+cin; 	t[7]=s;		
								
		s=t[4]+c;
		c=(s<c);
		t[4]=s;
		t[4+1]=c;
				
		m=minv*t[0];//????p??minv=1???????m=t[0]
		c=h_Hi64(m,Pa0);
		s=m*Pa0+t[0];   //??????????????????Pa0=-1, minv=1??????s=0;		
		c+=(s<t[0]);
		
		s=m*Pa1; c+=t[1]; cin=(c<t[1]); s+=c;   cin+=(s<c); c=h_Hi64(m,Pa1)+cin; t[0]=s;		
		s=m*Pa2; c+=t[2]; cin=(c<t[2]); s+=c;   cin+=(s<c); c=h_Hi64(m,Pa2)+cin; t[1]=s;
		s=m*Pa3; c+=t[3]; cin=(c<t[3]); s+=c;   cin+=(s<c); c=h_Hi64(m,Pa3)+cin; t[2]=s;		
	    // s=m*Pa4; c+=t[4]; cin=(c<t[4]); s+=c;   cin+=(s<c); c=h_Hi64(m,Pa4)+cin; t[3]=s;		
		// s=m*Pa5; c+=t[5]; cin=(c<t[5]); s+=c;   cin+=(s<c); c=h_Hi64(m,Pa5)+cin; t[4]=s;
		// s=m*Pa6; c+=t[6]; cin=(c<t[6]); s+=c;	cin+=(s<c); c=h_Hi64(m,Pa6)+cin; t[5]=s;		
		// s=m*Pa7; c+=t[7]; cin=(c<t[7]); s+=c;   cin+=(s<c); c=h_Hi64(m,Pa7)+cin; t[6]=s;
		
		s=t[4]+c;
		c=(s<c);
		t[4-1]=s;
		t[4]=t[4+1]+c;
	}
	
	j=(t[4]!=0);
	if(j==0)
	{
		for(i=4-1;i>=0;i--)
		{
			if(t[i]!=Pa[i])
			{				
				j=(t[i]>Pa[i]);
				break;
			}
			else if(i==0)//????,??t=P, ??????????0????
			{
				j=1;
			}
		}
	}
	
	//????	
	if(j)
	{
		cin=1;	
		for(i=0;i<4;i++)
		{
			m=cin+(~Pa[i]);
			cin=(m<cin);
			m+=t[i];
			cin+=(m<t[i]);
			Ca[i]=m;
		}
	}
	else
	{
		for(i=0;i<4;i++)
		{
			Ca[i]=t[i];
		}
	}
	
	#ifdef __CUDA_ARCH__	
		#undef h_Hi64 
	#endif	
}

__device__ __host__  void dh_mybig_monmult_64_modN(const UINT64 *Aa, const UINT64 *Ba ,UINT64 *Ca)//pass, c=a*b
{
	UINT64 t[4+2]={0};//8??64????????,???512
	
	//minv*P[0] mod 2^wordlen = -1. ????��????????64?????P[0]=0xffffffffffffffff=-1??????minv=1, minv*P[0]=-1 mod 2^64=-1
	
	//?????????P?????????????????��???????????????????????????
	// UINT64 minv=1;//minv????????????P?????��P[0]????2^64=-1(???P[0]??64????????????32?????????????2^32=-1??????? ????????????????????????2^wordlen?????????
	UINT64 minv = 0xc2e1f593efffffff;
	UINT64 m;	
	UINT64 c,s,cin;
	int i,j;
	
	#ifdef __CUDA_ARCH__	
		UINT64 *Pa=dc_N;	
		#define h_Hi64 __umul64hi
	#else
		const UINT64 *Pa=h_N;
	#endif	
	// const UINT64 *Pa=P; 
	
	for(i=0;i<4;i++)
	{
		c=0;
		m=Ba[i];
		
		s=m*Aa[0]; c+=t[0];	cin=(c<t[0]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[0])+cin; 	t[0]=s;
		s=m*Aa[1]; c+=t[1];	cin=(c<t[1]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[1])+cin; 	t[1]=s;
		s=m*Aa[2]; c+=t[2];	cin=(c<t[2]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[2])+cin; 	t[2]=s;
		s=m*Aa[3]; c+=t[3];	cin=(c<t[3]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[3])+cin; 	t[3]=s;
		// s=m*Aa[4]; c+=t[4];	cin=(c<t[4]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[4])+cin; 	t[4]=s;
		// s=m*Aa[5]; c+=t[5];	cin=(c<t[5]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[5])+cin; 	t[5]=s;
		// s=m*Aa[6]; c+=t[6];	cin=(c<t[6]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[6])+cin; 	t[6]=s;
		// s=m*Aa[7]; c+=t[7];	cin=(c<t[7]);		s+=c;		cin+=(s<c);	c=h_Hi64(m,Aa[7])+cin; 	t[7]=s;		
								
		s=t[4]+c;
		c=(s<c);
		t[4]=s;
		t[4+1]=c;
				
		m=minv*t[0];//????p??minv=1???????m=t[0]
		c=h_Hi64(m,Pa[0]);
		s=m*Pa[0]+t[0];   //??????????????????Pa0=-1, minv=1??????s=0;		
		c+=(s<t[0]);
		
		s=m*Pa[1]; c+=t[1]; cin=(c<t[1]); s+=c;   cin+=(s<c); c=h_Hi64(m,Pa[1])+cin; t[0]=s;		
		s=m*Pa[2]; c+=t[2]; cin=(c<t[2]); s+=c;   cin+=(s<c); c=h_Hi64(m,Pa[2])+cin; t[1]=s;
		s=m*Pa[3]; c+=t[3]; cin=(c<t[3]); s+=c;   cin+=(s<c); c=h_Hi64(m,Pa[3])+cin; t[2]=s;		
	    // s=m*Pa4; c+=t[4]; cin=(c<t[4]); s+=c;   cin+=(s<c); c=h_Hi64(m,Pa4)+cin; t[3]=s;		
		// s=m*Pa5; c+=t[5]; cin=(c<t[5]); s+=c;   cin+=(s<c); c=h_Hi64(m,Pa5)+cin; t[4]=s;
		// s=m*Pa6; c+=t[6]; cin=(c<t[6]); s+=c;	cin+=(s<c); c=h_Hi64(m,Pa6)+cin; t[5]=s;		
		// s=m*Pa7; c+=t[7]; cin=(c<t[7]); s+=c;   cin+=(s<c); c=h_Hi64(m,Pa7)+cin; t[6]=s;
		
		s=t[4]+c;
		c=(s<c);
		t[4-1]=s;
		t[4]=t[4+1]+c;
	}
	
	j=(t[4]!=0);
	if(j==0)
	{
		for(i=4-1;i>=0;i--)
		{
			if(t[i]!=Pa[i])
			{				
				j=(t[i]>Pa[i]);
				break;
			}
			else if(i==0)//????,??t=P, ??????????0????
			{
				j=1;
			}
		}
	}
	
	//????	
	if(j)
	{
		cin=1;	
		for(i=0;i<4;i++)
		{
			m=cin+(~Pa[i]);
			cin=(m<cin);
			m+=t[i];
			cin+=(m<t[i]);
			Ca[i]=m;
		}
	}
	else
	{
		for(i=0;i<4;i++)
		{
			Ca[i]=t[i];
		}
	}
	
	#ifdef __CUDA_ARCH__	
		#undef h_Hi64 
	#endif	
}




//////////////////////////////////////test inv//////////////////////////////////////
//A!=0 return 0
//A=0 return 1
__device__ __host__ int dh_mybig_iszero_64(const UINT64 *A)
{
	int i;
	for(i=0;i<4;i++)
	{
		if(A[i]!=0)	return 0;
	}
	return 1;	
}


//????��
//return 1	A>B
//return 0	A=B
//return -1 A<B
__device__ __host__ int dh_mybig_compare_64(const UINT64 *A, const UINT64 *B)
{
	int i;
	int flag=0;

	for(i=3;i>=0;i--)
	{
		if(A[i]>B[i])
		{
			flag=1;
			break;
		}
		else if(A[i]<B[i])
		{
			flag=-1;
			break;
		}
	}
	return (flag);
}

//C=A/2
__device__ __host__ void dh_mybig_half_64(const UINT64 *A, UINT64 *C)
{
	int i;
	UINT64 c,c1;

	c=0;
	for(i=4-1;i>=0;i--)
	{
		c1=A[i]&0x1;
		C[i]=c<<(63)|A[i]>>1;
		c=c1;
	}
}

//C=2A
__device__ __host__ UINT64 dh_mybig_double_64(const UINT64 *A, UINT64 *C)//pass
{
	int i;
	UINT64 c,c1;

	c=0;
	for(i=0;i<4;i++)
	{
		c1=A[i]>>63;
		C[i]=(C[i]<<1)|c;
		c=c1;
	}
	return c;
}

//C=A-B
__device__ __host__ void dh_mybig_sub_64(const UINT64 *A, const UINT64 *B, UINT64 *C)
{
	int i;
	UINT64 c,l,h;

	c=1;
	for(i=0;i<4;i++)
	{
		l=(~B[i])+c;
		h=(l<c);
		l+=A[i];
		c=h+(l<A[i]);
		C[i]=l;	
	}
}

//C=A+B
__device__ __host__ UINT64 dh_mybig_add_64(const UINT64 *A, const UINT64 *B, UINT64 *C)
{
	int i;
	UINT64 c,l,h;
	
	c=0;
	for(i=0;i<4;i++)
	{
		l=A[i]+c;
		h=(l<c);
		l+=B[i];
		c=h+(l<B[i]);
		C[i]=l;
	}
	return c;
}

/////////////////////////////////////////
//??2?
//????:		A P
//???:		C
//C=2*A mod P
//////////////////////////////////////////
__device__ __host__ void dh_mybig_moddouble_64(const UINT64 *A, const UINT64 *P, UINT64 *C)
{
	int i,sub_en=0;
	UINT64 cin,c,temp64;

	//??��
	cin=(A[0]>>63)&0x1;//????????��,?????2
	C[0]=A[0]<<1;
	for(i=1;i<4;i++)
	{
		c=(A[i]>>63)&0x1;
		C[i]=(A[i]<<1)|cin;
		cin=c;
	}

	//????��
	if(cin==1)//??????????��?1,????????????????��??2*A?????2*A?????p??
	{
		sub_en=1;
	}
	else
	{
		for(i=3;i>=0;i--)
		{
			if(C[i]!=P[i])
			{
				if(C[i]>P[i]) sub_en=1;
				break;
			}
			else if(i==0) sub_en=1;//????????
		}
	}
	
	//????
	if(sub_en)//???????��?if(sub_en!=0)???????????
	{
		cin=1;
		for(i=0;i<4;i++)
		{			
			temp64=(~P[i])+cin;
			c=(temp64<cin);
			temp64+=C[i];
			cin=c+(temp64<C[i]);
			C[i]=temp64;
		}
	}
}


__device__ __host__  int dh_mybig_iszero(const UINT64 * A)
{
	// if( A[0]|A[1]|A[2]|A[3]|A[4]|A[5]|A[6]|A[7])
	// 	return 0;
	// else return 1;
	if( A[0]|A[1]|A[2]|A[3])
		return 0;
	else return 1;
}

void h_mybig_print(const UINT64 *a)
{
	int i;
	unsigned char *t=(unsigned char*) a;
	for(i=32-1;i>=0;i--) if(*(t+i)) break;//??0???????
	if(i<0) printf("0");
	else
	{
		printf("%x",*(t+i)&0xff);//?????0?????
		for(i=i-1;i>=0;i--)printf("%02x",*(t+i)&0xff);
	}
	printf("\n");	
}
__device__ void d_mybig_print(const UINT64 *a)
{
	int i;
	unsigned char *t=(unsigned char*) a;
	for(i=32-1;i>=0;i--) if(*(t+i)) break;//??0???????
	if(i<0) printf("0");
	else
	{
		printf("%x",*(t+i)&0xff);//?????0?????
		for(i=i-1;i>=0;i--)printf("%02x",*(t+i)&0xff);
	}
	printf("\n");	
}

/////////////////////////////////////////
//??????: ???????C=A^-512 * 2^512 mod P
//????:		A C l(l???????????512)
//???:		C (C=A^-1 * 2^512 mod P)
//???: A<P
//////////////////////////////////////////
__device__ __host__ void dh_mybig_moninv(const UINT64 *A,UINT64 *C)//test
{
	int i,k;
	UINT64 U[4],V[4],R[4],S[4];
	int z,cp,cs,cr,sh;
	/*
	#ifdef __CUDA_ARCH__	
		UINT64 *P=dc_p;	
	#else
		const UINT64 *P=h_p;
	#endif	
	//*/

	/************************??********************************
	Preferences: E. Savas, C.K.Koc, The Montgomery Modular
	Inverse-Revisited, IEEE TRANSACTIONS ON COMPUTERS,
	VOL. 49, NO.7, JULY 2000

	--Phase I
	U=P V=A R=0 S=1
	k=0
	while(V>0)
		if U is even then U=U/2 S=2S
		else if V is even then V=V/2 R=2R
		else if U>V then U=(U-V)/2 R=R+S S=2S
		else if V>=U then V=(V-U)/2 S=S+R R=2R
		k=k+1
	if R>=P then R=R-P
	R=P-R

	--Phase II
  R=Mont(R*mR)=A^-1 * 2^k *  2^(2l-k) * 2^(-l) mod P 
  = A^-1 * 2^l mod P
	
	return R
	*************************************************************/

	////0. initial
	//U=P V=A R=0 S=1
	// U[0]=0xffffffffffffffffL; //U=P??P???????????????
	// U[1]=0xfffffffeffffffffL;
	// U[2]=0xffffffffffffffffL;
	// U[3]=0xfffffffeffffffffL;
	// U[4]=0xffffffffffffffffL;
	// U[5]=0xfffffffffffffffeL;
	// U[6]=0xfffffffeffffffffL;
	// U[7]=0xffffffffffffffffL;
	U[0] = Pa0;
	U[1] = Pa1;
	U[2] = Pa2;
	U[3] = Pa3;


	for(i=0;i<4;i++) V[i]=A[i];
	for(i=0;i<4;i++) R[i]=0;
	for(i=1;i<4;i++) S[i]=0;
	S[0]=0x1UL; 

	////1. phase I
	//get R=A^-1*2^k mod P
	k=0;
	z=dh_mybig_iszero_64(V);
	while(z==0)
	{	
		//printf("here k=%d  ",k);//getchar();
		cp=dh_mybig_compare_64(U,V);
		if((U[0]&0x1)==0)
		{
			dh_mybig_half_64(U,U);
			cs=dh_mybig_double_64(S,S);
		}
		else if((V[0]&0x1)==0)
		{			
			dh_mybig_half_64(V,V);
			cr=dh_mybig_double_64(R,R);
		}
		else if(cp==1)
		{
			dh_mybig_sub_64(U,V,U);
			dh_mybig_half_64(U,U);
			cr=dh_mybig_add_64(R,S,R);
			cs=dh_mybig_double_64(S,S);
		}
		else if(cp<1) //if(cp==0 || cp==-1)
		{
			dh_mybig_sub_64(V,U,V);
			dh_mybig_half_64(V,V);
			cs=dh_mybig_add_64(S,R,S);
			cr=dh_mybig_double_64(R,R);
		}
		k++;
		z=dh_mybig_iszero_64(V);
		//if(cr==1) printf("\nr\n");
		//if(cs==1) printf("\ns\n");		
	}
	
	// U[0]=0xffffffffffffffffL;
	// U[1]=0xfffffffeffffffffL;
	// U[2]=0xffffffffffffffffL;
	// U[3]=0xfffffffeffffffffL;
	// U[4]=0xffffffffffffffffL;
	// U[5]=0xfffffffffffffffeL;
	// U[6]=0xfffffffeffffffffL;
	// U[7]=0xffffffffffffffffL;
	U[0] = Pa0;
	U[1] = Pa1;
	U[2] = Pa2;
	U[3] = Pa3;

	cp=dh_mybig_compare_64(R,U);
	if(cp==1 || cp==0 || cr==1)
	{
		dh_mybig_sub_64(R,U,R);
	}
	dh_mybig_sub_64(U,R,R);
	////////////////////
	//the result of phase I is R
	//R=A^-1 * 2^k mod P

    
	////2. phaseII
	//get R=A^-1 * 2^l mod P
 	// R=Mont(R*mR)=A^-1 * 2^k *  2^(2l-k) * 2^(-l) mod P = A^-1 * 2^l mod P
 	 if(k>256)
	 {	
		// printf("case 1\n");
	    for(i=0;i<4;i++) V[i]=0;
	    V[(int)((512-k)/64)]=(((UINT64)1)<<((int)((512-k)%64)));
	    dh_mybig_monmult_64(R,V,C);
	 }	
	 else if(k==256) 
	 {
			// printf("case 2\n");
			// V[0]=0x0000000000000001L;
			// V[1]=0x0000000100000000L;
			// V[2]=0x0000000000000000L;
			// V[3]=0x0000000100000000L;
			// V[4]=0x0000000000000000L;
			// V[5]=0x0000000000000001L;
			// V[6]=0x0000000100000000L;
			// V[7]=0x0000000000000000L;
			// dh_mybig_monmult_64(R,V,C);		
			for(i=0;i<4;i++) C[i]=R[i];	
		}
		else if(k<256) 
		{
		//   printf("case 3\n");
		  for(i=1;i<=(256-k);i++)
			{		
				dh_mybig_moddouble_64(R,U,R);			
			}
		  for(i=0;i<4;i++) C[i]=R[i];
		}

}

/////////////////////////////////////////
//??????: ???????C=A^-512 * 2^512 mod P ???????P
//????:		A C l(l???????????512)
//???:		C (C=A^-1 * 2^512 mod P)
//???: A<P
//////////////////////////////////////////
__device__ __host__ void dh_mybig_moninv_modN(const UINT64 *A,UINT64 *C)//test
{
	int i,k;
	UINT64 U[4],V[4],R[4],S[4];
	int z,cp,cs,cr,sh;
	/*
	#ifdef __CUDA_ARCH__	
		UINT64 *P=dc_p;	
	#else
		const UINT64 *P=h_p;
	#endif	
	//*/

	/************************??********************************
	Preferences: E. Savas, C.K.Koc, The Montgomery Modular
	Inverse-Revisited, IEEE TRANSACTIONS ON COMPUTERS,
	VOL. 49, NO.7, JULY 2000

	--Phase I
	U=P V=A R=0 S=1
	k=0
	while(V>0)
		if U is even then U=U/2 S=2S
		else if V is even then V=V/2 R=2R
		else if U>V then U=(U-V)/2 R=R+S S=2S
		else if V>=U then V=(V-U)/2 S=S+R R=2R
		k=k+1
	if R>=P then R=R-P
	R=P-R

	--Phase II
  R=Mont(R*mR)=A^-1 * 2^k *  2^(2l-k) * 2^(-l) mod P 
  = A^-1 * 2^l mod P
	
	return R
	*************************************************************/

	////0. initial
	//U=P V=A R=0 S=1
	// U[0]=0xffffffffffffffffL; //U=P??P???????????????
	// U[1]=0xfffffffeffffffffL;
	// U[2]=0xffffffffffffffffL;
	// U[3]=0xfffffffeffffffffL;
	// U[4]=0xffffffffffffffffL;
	// U[5]=0xfffffffffffffffeL;
	// U[6]=0xfffffffeffffffffL;
	// U[7]=0xffffffffffffffffL;
	U[0] = 0xBFD25E8CD0364141L;
	U[1] = 0xBAAEDCE6AF48A03BL;
	U[2] = 0xFFFFFFFFFFFFFFFEL;
	U[3] = 0xFFFFFFFFFFFFFFFFL;


	for(i=0;i<4;i++) V[i]=A[i];
	for(i=0;i<4;i++) R[i]=0;
	for(i=1;i<4;i++) S[i]=0;
	S[0]=0x1UL; 

	////1. phase I
	//get R=A^-1*2^k mod P
	k=0;
	z=dh_mybig_iszero_64(V);
	while(z==0)
	{	
		//printf("here k=%d  ",k);//getchar();
		cp=dh_mybig_compare_64(U,V);
		if((U[0]&0x1)==0)
		{
			dh_mybig_half_64(U,U);
			cs=dh_mybig_double_64(S,S);
		}
		else if((V[0]&0x1)==0)
		{			
			dh_mybig_half_64(V,V);
			cr=dh_mybig_double_64(R,R);
		}
		else if(cp==1)
		{
			dh_mybig_sub_64(U,V,U);
			dh_mybig_half_64(U,U);
			cr=dh_mybig_add_64(R,S,R);
			cs=dh_mybig_double_64(S,S);
		}
		else if(cp<1) //if(cp==0 || cp==-1)
		{
			dh_mybig_sub_64(V,U,V);
			dh_mybig_half_64(V,V);
			cs=dh_mybig_add_64(S,R,S);
			cr=dh_mybig_double_64(R,R);
		}
		k++;
		z=dh_mybig_iszero_64(V);
		//if(cr==1) printf("\nr\n");
		//if(cs==1) printf("\ns\n");		
	}
	
	// U[0]=0xffffffffffffffffL;
	// U[1]=0xfffffffeffffffffL;
	// U[2]=0xffffffffffffffffL;
	// U[3]=0xfffffffeffffffffL;
	// U[4]=0xffffffffffffffffL;
	// U[5]=0xfffffffffffffffeL;
	// U[6]=0xfffffffeffffffffL;
	// U[7]=0xffffffffffffffffL;
	U[0] = 0xBFD25E8CD0364141L;
	U[1] = 0xBAAEDCE6AF48A03BL;
	U[2] = 0xFFFFFFFFFFFFFFFEL;
	U[3] = 0xFFFFFFFFFFFFFFFFL;

	cp=dh_mybig_compare_64(R,U);
	if(cp==1 || cp==0 || cr==1)
	{
		dh_mybig_sub_64(R,U,R);
	}
	dh_mybig_sub_64(U,R,R);
	////////////////////
	//the result of phase I is R
	//R=A^-1 * 2^k mod P

    
	////2. phaseII
	//get R=A^-1 * 2^l mod P
 	// R=Mont(R*mR)=A^-1 * 2^k *  2^(2l-k) * 2^(-l) mod P = A^-1 * 2^l mod P
 	 if(k>256)
	 {	
		// printf("case 1\n");
	    for(i=0;i<4;i++) V[i]=0;
	    V[(int)((512-k)/64)]=(((UINT64)1)<<((int)((512-k)%64)));
	    dh_mybig_monmult_64_modN(R,V,C);
	 }	
	 else if(k==256) 
	 {
			// printf("case 2\n");
			// V[0]=0x0000000000000001L;
			// V[1]=0x0000000100000000L;
			// V[2]=0x0000000000000000L;
			// V[3]=0x0000000100000000L;
			// V[4]=0x0000000000000000L;
			// V[5]=0x0000000000000001L;
			// V[6]=0x0000000100000000L;
			// V[7]=0x0000000000000000L;
			// dh_mybig_monmult_64(R,V,C);		
			for(i=0;i<4;i++) C[i]=R[i];	
		}
		else if(k<256) 
		{
		//   printf("case 3\n");
		  for(i=1;i<=(256-k);i++)
			{		
				dh_mybig_moddouble_64(R,U,R);			
			}
		  for(i=0;i<4;i++) C[i]=R[i];
		}

}

//GAO: mod exp
//C = A^k mod P
__device__ __host__ void dh_mybig_modexp(UINT64* a,UINT64 *k,UINT64* c){
	//gyy
	int i,j;
	UINT64 tbn[4];
	// Jpoint t2;
	

	// find first 1
	for(i=3;i>=0;i--){
		if(k[i]!=0)	break;
	}
	if(i<0){
		printf("mod exp:k==0!!!!!!!!!\n");
	}
	for(j=63;j>=0;j--){
		if(((k[i]>>j)&0x01)!=0) break;
	}
	
	dh_mybig_copy(tbn,a);

	j--;


		
		for(;j>=0;j--){
				// printf("double\n");
			dh_mybig_monmult_64(tbn,tbn,tbn);
			// ppoint_double(pt1,pt1);
			if(((k[i]>>j)&0x01)==1){
					// printf("add\n");
				dh_mybig_monmult_64(tbn,a,tbn);
				// dh_ellipticAdd_JJ(pt1,&tp,pt1);
			}
		}
		i--;
		for(;i>=0;i--){
			for(j=63;j>=0;j--){
					// printf("double\n");
				// ppoint_double(pt1,pt1);
				dh_mybig_monmult_64(tbn,tbn,tbn);
				if(((k[i]>>j)&0x01)==1){
						// printf("add\n");
					// dh_ellipticAdd_JJ(pt1,&tp,pt1);
					dh_mybig_monmult_64(tbn,a,tbn);
				}
			}
		}

		dh_mybig_copy(c,tbn);
	
	// printf("j=%d,i=%d\n",j,i);
	
	// printf("copy\n");
	// dh_mybig_copy(pt1->x,tp.x);
	// dh_mybig_copy(pt1->y,tp.y);
	// dh_mybig_copy(pt1->z,tp.z);
}

__device__ __host__ void dh_mybig_modexp_modN(UINT64* a,UINT64 *k,UINT64* c){
	//gyy
	int i,j;
	UINT64 tbn[4];
	// Jpoint t2;
	

	// find first 1
	for(i=3;i>=0;i--){
		if(k[i]!=0)	break;
	}
	if(i<0){
		printf("mod exp:k==0!!!!!!!!!\n");
	}
	for(j=63;j>=0;j--){
		if(((k[i]>>j)&0x01)!=0) break;
	}
	
	dh_mybig_copy(tbn,a);

	j--;


		
		for(;j>=0;j--){
				// printf("double\n");
			dh_mybig_monmult_64_modN(tbn,tbn,tbn);
			// ppoint_double(pt1,pt1);
			if(((k[i]>>j)&0x01)==1){
					// printf("add\n");
				dh_mybig_monmult_64_modN(tbn,a,tbn);
				// dh_ellipticAdd_JJ(pt1,&tp,pt1);
			}
		}
		i--;
		for(;i>=0;i--){
			for(j=63;j>=0;j--){
					// printf("double\n");
				// ppoint_double(pt1,pt1);
				dh_mybig_monmult_64_modN(tbn,tbn,tbn);
				if(((k[i]>>j)&0x01)==1){
						// printf("add\n");
					// dh_ellipticAdd_JJ(pt1,&tp,pt1);
					dh_mybig_monmult_64_modN(tbn,a,tbn);
				}
			}
		}

		dh_mybig_copy(c,tbn);
	
	// printf("j=%d,i=%d\n",j,i);
	
	// printf("copy\n");
	// dh_mybig_copy(pt1->x,tp.x);
	// dh_mybig_copy(pt1->y,tp.y);
	// dh_mybig_copy(pt1->z,tp.z);
}

__device__  void dh_mybig_modexp_ui32_modN(UINT64* a,unsigned int k,UINT64* c){
	//gyy
	if(k==0){
		dh_mybig_copy(c,dc_mon_ONE_modN);
		return;
	}
	int i;
	UINT64 tbn[4];
	// Jpoint t2;
	
	for(i=31;i>=0;i--){
		if(((k>>i)&0x01)!=0) break;
	}
	// find first 1
	
	// if(i<0){
	// 	printf("mod exp:k==0!!!!!!!!!\n");
	// }
	// for(j=63;j>=0;j--){
	// 	if((k[i]>>j)&0x01!=0) break;
	// }
	
	dh_mybig_copy(tbn,a);

	i--;

	for(;i>=0;i--){
		dh_mybig_monmult_64_modN(tbn,tbn,tbn);
		if(((k>>i)&0x01)==1){
			dh_mybig_monmult_64_modN(tbn,a,tbn);
		}
	}


	dh_mybig_copy(c,tbn);
	
	// printf("j=%d,i=%d\n",j,i);
	
	// printf("copy\n");
	// dh_mybig_copy(pt1->x,tp.x);
	// dh_mybig_copy(pt1->y,tp.y);
	// dh_mybig_copy(pt1->z,tp.z);
}

//GAO: mod exp
//C = 2^k1 * 3^k2 mod P
__device__ __host__ void dh_mybig_modexp23_modN(unsigned int k2, unsigned int k3, UINT64* c){
    //occulticplus
    int i,j;
    UINT64 tbn[4] = {0};

    UINT64 pow2[34] = {0x1L, 0x2L, 0x4L, 0x8L, 0x10L, 0x20L, 0x40L, 0x80L, 0x100L, 0x200L, 0x400L, 0x800L,
                   0x1000L, 0x2000L, 0x4000L, 0x8000L, 0x10000L, 0x20000L, 0x40000L, 0x80000L,
                   0x100000L, 0x200000L, 0x400000L, 0x800000L, 0x10000000L, 0x2000000L, 0x4000000L, 0x8000000L,
                   0x10000000L, 0x20000000L, 0x40000000L, 0x80000000L, 0x1000000000L};
    UINT64 pow3[41] = {1L, 3L, 9L, 27L, 81L, 243L, 729L, 2187L, 6561L, 19683L, 59049L, 177147L, 531441L, 1594323L,
                       4782969L, 14348907L, 43046721L, 129140163L, 387420489L, 1162261467L, 3486784401L, 10460353203L,
                       31381059609L, 94143178827L, 282429536481L, 847288609443L, 2541865828329L, 7625597484987L,
                       22876792454961L, 68630377364883L, 205891132094649L, 617673396283947L, 1853020188851841L,
                       5559060566555523L, 16677181699666569L, 50031545098999707L, 150094635296999121L,
                       450283905890997363L, 1350851717672992089L, 4052555153018976267L};
    // Jpoint t2;

    int base2 = k2 && ((1 << 6) - 1);
    int id2 = k2 >> 6;
    assert(id2 <= 3);
    tbn[id2] |= 1 << base2;
    int base3 = k3 && ((1 << 6) - 1);
    int id3 = k3 >> 6;
    UINT64 dot[4] = {pow3[32]};
    for (int i = 0; i < base3; i++) {
        dh_mybig_monmult_64(tbn, tbn, dot); // no definition;
    }
    dot[0] = pow3[id3];
    dh_mybig_monmult_64(tbn, tbn, dot);

    dh_mybig_copy(c,tbn);

    // printf("j=%d,i=%d\n",j,i);

    // printf("copy\n");
    // dh_mybig_copy(pt1->x,tp.x);
    // dh_mybig_copy(pt1->y,tp.y);
    // dh_mybig_copy(pt1->z,tp.z);
}

//////////////////////////////////////test inv end//////////////////////////////////////

/////////////////GPU????????????????////////////////////////////////////////////////////


/////////////////GPU?????????????????��?????????????????????????????????????????????///////////////////////////////////////
//GAO:��???
__device__ __host__  void dh_setzero_J(Jpoint *pt)
{
	int i;
	for(i=0;i<8;i++) pt->z[i]=0UL;
}
//GAO:��???
__device__ __host__  void dh_setzero_A(Apoint *pt)
{
	int i;
	for(i=0;i<8;i++) pt->x[i]=0UL;	
}

__device__ __host__  int dh_iszero_J(const Jpoint *pt)
{
	// if((pt->z[0]|pt->z[1]|pt->z[2]|pt->z[3]|pt->z[4]|pt->z[5]|pt->z[6]|pt->z[7]) == 0UL)	return 1;
	// return 0;	
	if((pt->z[0]|pt->z[1]|pt->z[2]|pt->z[3]) == 0UL)	return 1;
	return 0;
}


__device__ __host__  int dh_iszero_A(const Apoint *pt)
{	
	// if((pt->x[0]|pt->x[1]|pt->x[2]|pt->x[3]|pt->x[4]|pt->x[5]|pt->x[6]|pt->x[7]) == 0UL)	return 1;
	// return 0;
	if((pt->x[0]|pt->x[1]|pt->x[2]|pt->x[3]) == 0UL)	return 1;
	return 0;
}

//GAO:��???
//???????????????????????
__device__ __host__  void dh_normlize_J(Jpoint *pt)
{	
	UINT64 tmp[8],invtmp[8];
	
	if(dh_iszero_J(pt))	return;
	
	tmp[0]=0x1L; tmp[1]=0x0L; tmp[2]=0x0L; tmp[3]=0x0L; tmp[4]=0x0L; tmp[5]=0x0L; tmp[6]=0x0L; tmp[7]=0x0L;
	
	
	dh_mybig_monmult_64(pt->z,tmp, invtmp);
	dh_mybig_moninv(invtmp, invtmp);
	//invMod(invtmp,pt->z,mod);//invtmp=1/z
	
	//squareMod(tmp,invtmp,mod);//tmp=1/zz
	dh_mybig_monmult_64(invtmp,invtmp, tmp);
	
	//productMod(pt->x,pt->x,tmp,mod);//x/zz
	dh_mybig_monmult_64(pt->x,tmp, pt->x);		
	
	//productMod(tmp,tmp,invtmp,mod);//1/zzz
	dh_mybig_monmult_64(tmp,invtmp, tmp);
	
	//productMod(pt->y,pt->y,tmp,mod);//y/zzz
	dh_mybig_monmult_64(pt->y,tmp, pt->y);
	
	//z=mon_one
	memcpy(pt->z, dc_mon_ONE, sizeof(pt->z));
	// pt->z[0]=0x1000003d1L;
	// pt->z[1]=0x0000000100000000L;
	// pt->z[2]=0x0000000000000000L;
	// pt->z[3]=0x0000000100000000L;
	// pt->z[4]=0x0000000000000000L;
	// pt->z[5]=0x0000000000000001L;
	// pt->z[6]=0x0000000100000000L;
	// pt->z[7]=0x0000000000000000L;	
	
}
//GAO:��???
__device__ void d_mon2normal_J(Jpoint *pt)
{
	dh_mybig_monmult_64(pt->x, dc_ONE, pt->x);
	dh_mybig_monmult_64(pt->y, dc_ONE, pt->y);
	dh_mybig_monmult_64(pt->z, dc_ONE, pt->z);	
}
//GAO:��???
__device__ __host__  void dh_mon2normal_A(Apoint *pt)
{
	#ifdef __CUDA_ARCH__
	dh_mybig_monmult_64(pt->x, dc_ONE, pt->x);
	dh_mybig_monmult_64(pt->y, dc_ONE, pt->y);
	#else
	dh_mybig_monmult_64(pt->x, h_ONE, pt->x);
	dh_mybig_monmult_64(pt->y, h_ONE, pt->y);
	#endif
}

void h_print_pointA(const Apoint *pt)
{
	if(dh_iszero_A(pt))
	{ 
		printf("(Infinity)\n");
		return;
	}	
	printf("x: ");h_mybig_print(pt->x);
	printf("y: ");h_mybig_print(pt->y);
	
}
 
void h_print_pointJ(const Jpoint *pt)
{
	/*
	if(dh_iszero_J(pt))
	{ 
		printf("(Infinity)\n");
		return;
	}*/
	printf("x: ");h_mybig_print(pt->x);
	printf("y: ");h_mybig_print(pt->y);
	printf("z: ");h_mybig_print(pt->z);
}

void __device__ d_print_pointJ(const Jpoint *pt)
{
	d_mybig_print(pt->x);printf("\n");
	d_mybig_print(pt->y);printf("\n");
	d_mybig_print(pt->z);printf("\n");
}

__device__ __host__ void ppoint_double(Jpoint *pt1,Jpoint* pt2){
	UINT64 u1[4],u2[4],u3[4];

	#ifdef __CUDA_ARCH__
		const UINT64 *Pa=dc_p;
	#else
		const UINT64 *Pa=h_p;
	#endif	

	//?????��??????????
	//secp256k1??a?0??????????????????????????
	dh_mybig_moddouble_64(pt1->y,Pa,u1); 	//u1=2y
	dh_mybig_monmult_64(pt1->z,u1,pt2->z);		//z=u1*z=2yz

	dh_mybig_monmult_64(pt1->x,pt1->x,u2);	//u2=x^2
	dh_mybig_moddouble_64(u2,Pa,u3);		//u3=2*u2=2x^2
	dh_mybig_modadd_64(u3,u2,u3);			//u3=u3+u2 = 3x^2 = lambda_1
	dh_mybig_monmult_64(u1,pt1->y,u1);		//u1 = u1*y = 2y^2
	dh_mybig_monmult_64(u1,pt1->x,u2);		//u2 = u1*x = 2xy^2 
	dh_mybig_moddouble_64(u2,Pa,u2);		//u2 = 2*u2 = 4xy^2= lambda_2
	dh_mybig_monmult_64(u3,u3,pt2->x);		//pt1x = lambda_1^2
	dh_mybig_moddouble_64(u2,Pa,pt2->y);		//pt1y = 2*u2 = 2*lambda_2
	dh_mybig_modsub_64(pt2->x,pt2->y,pt2->x);	//x = pt1x-pt1y = lambda_1^2-2*labmda_2

	dh_mybig_monmult_64(u1,u1,u1);			//u1 = u1*u1 = 4y^4;
	dh_mybig_moddouble_64(u1,Pa,u1);		//u1 = 2u1 = 8y^4 = lambda_3

	dh_mybig_modsub_64(u2,pt2->x,u2);		//u2 = u2-pt1x = lambda2-pt1x
	dh_mybig_monmult_64(u3,u2,pt2->y);		//pt1y = u2*u3 = Lambda_1 * (lambda2-pt1x)
	dh_mybig_modsub_64(pt2->y,u1,pt2->y);	//y = pt1y - labmda_3;
}
//???????
__device__ __host__ void ppoint_double_V2(Jpoint *pt1,Jpoint* pt2){
	UINT64 u1[4],u2[4],u3[4];

	#ifdef __CUDA_ARCH__
		const UINT64 *Pa=dc_p;
	#else
		const UINT64 *Pa=h_p;
	#endif	

	dh_mybig_monmult_64(pt1->x,pt1->x,u2); 	//u2 = x^2
	dh_mybig_moddouble_64(u2,Pa,pt2->y);		//u1 = 2u2 = 2*x^2
	dh_mybig_modadd_64(pt2->y,u2,pt2->y);		//u1 = u1+u2 = 3*x^3 = lamb1
	dh_mybig_moddouble_64(u3,Pa,u3);		//p2y = 2y
	dh_mybig_monmult_64(pt2->z,u3,pt2->z); 	//p2y = 2yz
	
	dh_mybig_monmult_64(u3,u3,u3);  //p2y = 4y^2
	dh_mybig_monmult_64(pt1->x,u3,u2);		//u2 = 4xy^2

	dh_mybig_monmult_64(u3,u3,u3);  //p2y = 2py2 = 16y^4
	dh_mybig_monmult_64(u3,dc_mon_inv_two,u3);//p2y = p2y/2 = 8y^4

	dh_mybig_monmult_64(pt2->y,pt2->y,pt2->x);			//p2x = u1*u1 = lamb1^2
	dh_mybig_modsub_64(pt2->x,pt2->y,pt2->x);		//p2x = p2x-u2
	dh_mybig_modsub_64(pt2->x,pt2->y,pt2->x);		//p2x = p2x-u2 = lamb1^2-2lamb2

	dh_mybig_modsub_64(u2,pt2->x,u2);			//u2 = u2-p2x=lamb2-x
	dh_mybig_monmult_64(pt2->y,u2,pt2->y);

	dh_mybig_modsub_64(pt2->y,u3,pt2->y);
	//?????��??????????3
	//secp256k1??a?0??????????????????????????
	// dh_mybig_moddouble_64(pt1->y,Pa,u1); 	//u1=2y
	// dh_mybig_monmult_64(pt1->z,u1,pt2->z);		//z=u1*z=2yz

	// dh_mybig_monmult_64(pt1->x,pt1->x,u2);	//u2=x^2
	// dh_mybig_moddouble_64(u2,Pa,u3);		//u3=2*u2=2x^2
	// dh_mybig_modadd_64(u3,u2,u3);			//u3=u3+u2 = 3x^2 = lambda_1
	// dh_mybig_monmult_64(u1,pt1->y,u1);		//u1 = u1*y = 2y^2
	// dh_mybig_monmult_64(u1,pt1->x,u2);		//u2 = u1*x = 2xy^2 
	// dh_mybig_moddouble_64(u2,Pa,u2);		//u2 = 2*u2 = 4xy^2= lambda_2
	// dh_mybig_monmult_64(u3,u3,pt2->x);		//pt1x = lambda_1^2
	// dh_mybig_moddouble_64(u2,Pa,pt2->y);		//pt1y = 2*u2 = 2*lambda_2
	// dh_mybig_modsub_64(pt2->x,pt2->y,pt2->x);	//x = pt1x-pt1y = lambda_1^2-2*labmda_2

	// dh_mybig_monmult_64(u1,u1,u1);			//u1 = u1*u1 = 4y^4;
	// dh_mybig_moddouble_64(u1,Pa,u1);		//u1 = 2u1 = 8y^4 = lambda_3

	// dh_mybig_modsub_64(u2,pt2->x,u2);		//u2 = u2-pt1x = lambda2-pt1x
	// dh_mybig_monmult_64(u3,u2,pt2->y);		//pt1y = u2*u3 = Lambda_1 * (lambda2-pt1x)
	// dh_mybig_modsub_64(pt2->y,u1,pt2->y);	//y = pt1y - labmda_3;
}

/*
 * Point Triple Function.
 * function: pt2 = pt1^3 = 3 \dots pt1.
 * requirements: pt1, pt2 coordinates are represented in Montgomery Field.
 * Attention: pt2 can't be pt1.
 * */
__device__ __host__ void ppoint_triple(Jpoint *pt1,Jpoint* pt2){
    UINT64 h_mon_a[4]={0xfffffffffffffffcL,0xfffffffc00000003L,0xffffffffffffffffL,0xfffffffbffffffffL};
    UINT64 c[4], m[4], e[4], t[4], u[4], wtf[4], foo[4];
#ifdef __CUDA_ARCH__
    const UINT64 *Pa=dc_p;
#else
    const UINT64 *Pa=h_p;
#endif
	dh_mybig_monmult_64(pt1->x, pt1->x, pt2->x); //xx = x1^2
    dh_mybig_monmult_64(pt1->z, pt1->z, pt2->z); //zz = z1^2
    dh_mybig_monmult_64(pt1->y, pt1->y, pt2->y); //yy = y1^2
    dh_mybig_moddouble_64(pt2->y, Pa, c); //c = 2 * yy

    dh_mybig_monmult_64(pt1->x, pt1->x, pt2->x);
    dh_mybig_moddouble_64(pt2->x, Pa, m);
    dh_mybig_modadd_64(m, pt2->x, m);
    //dh_mybig_monmult_64(h_mon_a, pt2->z, wtf); // a must be in montgomery field.
    //dh_mybig_modadd_64(m, wtf, m);// m = 3 * x1^2 + a * zz^2

    dh_mybig_monmult_64(pt1->x, c, foo);
    dh_mybig_monmult_64(m, m, wtf);
    dh_mybig_moddouble_64(foo, Pa, e);
    dh_mybig_modadd_64(foo, e, foo);
    dh_mybig_moddouble_64(foo, Pa, e);
    dh_mybig_modsub_64(e, wtf, e);//e = 6 * x1 * c - M^2

    dh_mybig_monmult_64(c, c, wtf);
    dh_mybig_moddouble_64(wtf, Pa, t);//t = 2 * c^2

    dh_mybig_monmult_64(m, e, u);
    dh_mybig_modsub_64(u, t, u);//u = m*e - t

    dh_mybig_monmult_64(e, e, wtf);//ee = e^2, use wtf to store.

    dh_mybig_moddouble_64(u, Pa, foo);
    dh_mybig_moddouble_64(foo, Pa, foo);//u4 = 4 * u, use foo to store.

    dh_mybig_monmult_64(pt1->x, wtf, pt2->x);
    dh_mybig_monmult_64(c, foo, c);
    dh_mybig_modsub_64(pt2->x, c, pt2->x);//x3 = x1*ee - c * u4

    dh_mybig_modsub_64(t, u, t);
    dh_mybig_monmult_64(foo, t, u); // foo = u4 out of life, u = u4(t - u).
    dh_mybig_monmult_64(e, wtf, foo);
    dh_mybig_modsub_64(u, foo, wtf);
    dh_mybig_monmult_64(pt1->y, wtf, pt2->y);//y3 = y1*(u4*(t-u) - e*ee)

    dh_mybig_monmult_64(pt1->z, e, pt2->z);//z3 = z1*e

}

// 7m + 7s solution, for a != 0.
// i know why this is incorrect: it has nothing to do with a in sekp256k1(a=0 in sekp256k1)
__device__ __host__ void ppoint_triple_v2(Jpoint *pt1,Jpoint* pt2) {
    UINT64 h_mon_a[4] = {0xfffffffffffffffcL, 0xfffffffc00000003L, 0xffffffffffffffffL, 0xfffffffbffffffffL};
    UINT64 dy[4], dz[4], dm[4], de[4], qy[4], t[4], m[4], e[4], u[4], wtf[4];

#ifdef __CUDA_ARCH__
    const UINT64 *Pa=dc_p;
#else
    const UINT64 *Pa=h_p;
#endif
    dh_mybig_monmult_64(pt1->y, pt1->y, dy); // dy = y1^2
    dh_mybig_monmult_64(pt1->z, pt1->z, dz); // dz = z1^2
    dh_mybig_monmult_64(dy, dy, qy); // qy = dy^2

	dh_mybig_monmult_64(pt1->x, pt1->x, m);
	dh_mybig_moddouble_64(m, Pa, wtf); // wtf = 2 * dx1
	dh_mybig_modadd_64(m, wtf, m); // m = 3 * dx1
	dh_mybig_monmult_64(dz, dz, wtf);
	dh_mybig_monmult_64(wtf, h_mon_a, wtf);
    dh_mybig_modadd_64(m, wtf, m);
	// error! this is relevant to a. m = 3*x1^2 + a * dz^2

    dh_mybig_monmult_64(m, m, dm); // dm = m^2

    dh_mybig_monmult_64(pt1->x, dy, wtf);
    dh_mybig_moddouble_64(wtf, Pa, e);
    dh_mybig_modadd_64(e, wtf, e);
    dh_mybig_moddouble_64(e, Pa, wtf);
    dh_mybig_moddouble_64(wtf, Pa, e);
    dh_mybig_modsub_64(e, dm, e); // e = 12 * x1 * dy - dm

    dh_mybig_monmult_64(e, e, de); // de = e^2

    dh_mybig_moddouble_64(qy, Pa, wtf);
    dh_mybig_moddouble_64(wtf, Pa, t);
    dh_mybig_moddouble_64(t, Pa, wtf);
    dh_mybig_moddouble_64(wtf, Pa, t); //t = 16 * qy

    dh_mybig_modadd_64(m, e, wtf);
    dh_mybig_monmult_64(wtf, wtf, u);
    dh_mybig_modadd_64(t, dm, t);
    dh_mybig_modadd_64(t, de, t);
    dh_mybig_modsub_64(u, t, u); // u = (m + e)^2 - dm - de - t

    dh_mybig_monmult_64(dy, u, wtf);
    dh_mybig_moddouble_64(wtf, Pa, pt2->x);
    dh_mybig_moddouble_64(pt2->x, Pa, wtf); // wtf = 4 * dy * u
    dh_mybig_monmult_64(pt1->x, de, pt2->x);
    dh_mybig_modsub_64(pt2->x, wtf, pt2->x);
    dh_mybig_moddouble_64(pt2->x, Pa, wtf);
    dh_mybig_moddouble_64(wtf, Pa, pt2->x); // x3 = 4(x1 * de - 4 * dy * u)


    dh_mybig_modsub_64(t, u, wtf);
    dh_mybig_monmult_64(wtf, u, pt2->y); // store u * (t - u)
    dh_mybig_monmult_64(e, de, wtf);
    dh_mybig_modsub_64(pt2->y, wtf, pt2->y);
    dh_mybig_monmult_64(pt2->y, pt1->y, wtf); // wtf = y1 * (u * (t - u) - e * de)
    dh_mybig_moddouble_64(wtf, Pa, pt2->y);
    dh_mybig_moddouble_64(pt2->y, Pa, wtf);
    dh_mybig_moddouble_64(wtf, Pa, pt2->y);// y3 = 8 * y1 * (u * (t - u) - e * de)

    dh_mybig_modadd_64(pt1->z, e, wtf);
    dh_mybig_monmult_64(wtf, wtf, pt2->z);
    dh_mybig_modadd_64(de, dz, de);
    dh_mybig_modsub_64(pt2->z, de, pt2->z); // z3 = (z1 + e)^2 - dz - de


}


// ref: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl, note here "a" must be 0.
// ATTENTION: pt1's address should be different with pt2's, and pt1's VALUE SHOULD BE DIFFERENT WITH pt2's.
// shortly, this function can't be used to calculate doubling.
__device__ __host__ void dh_ellipticAdd_JJ(Jpoint *pt1, Jpoint* pt2,Jpoint* pt3){
	UINT64 u1[4],u2[4],u3[4],u4[4];
	//u1=z2^2

	#ifdef __CUDA_ARCH__
		const UINT64 *Pa=dc_p;
		const UINT64 *mon_inv_two = dc_mon_inv_two;

	#else
		const UINT64 *Pa=h_p;
		const UINT64 *mon_inv_two = h_mon_inv_two;

	#endif	

	dh_mybig_monmult_64(pt2->z,pt2->z,u1);

	

	//u2=z1^2
	dh_mybig_monmult_64(pt1->z,pt1->z,u2);


	//u3 = u1*x1=x1*z2^2=lam1
	dh_mybig_monmult_64(pt1->x,u1,u3);
	
	//u4 = u2*x2 = x2*z1^2=lam2
	dh_mybig_monmult_64(pt2->x,u2,u4);
	
	

	//u1 = z2^3
	dh_mybig_monmult_64(u1,pt2->z,u1);
	//u2 = z1^3
	dh_mybig_monmult_64(u2,pt1->z,u2);
	
	//u1 = u1*y1 = lam4
	dh_mybig_monmult_64(u1,pt1->y,u1);
	//u2 = u2*y2 = lam5
	dh_mybig_monmult_64(u2,pt2->y,u2);


	//pt3y = u3-u4 = lam3
	dh_mybig_modsub_64(u3,u4,pt3->y);

	

	//u3+=u4 = lam1+lam2 = lam7
	dh_mybig_modadd_64(u3,u4,u3);
	//U4=u1-u2=lam4-lam5=lam6
	dh_mybig_modsub_64(u1,u2,u4);

	
	//u1+=u2 = lam4+lam5=lam8
	dh_mybig_modadd_64(u1,u2,u1);

	

	//pt3z = z1*z2*lam3
	dh_mybig_monmult_64(pt1->z,pt2->z,pt3->z);
	dh_mybig_monmult_64(pt3->z,pt3->y,pt3->z);

	
	//pt3x = u4^2 = lam6^2
	dh_mybig_monmult_64(u4,u4,pt3->x);
	//u2 = pt3y^2 = lam3^2
	dh_mybig_monmult_64(pt3->y,pt3->y,u2);

	

	//pt3y = pt3y*u2 = lam3^3
	dh_mybig_monmult_64(pt3->y,u2,pt3->y);

	//u2 = u2*u3 = lam7*lam3^2
	dh_mybig_monmult_64(u2,u3,u2);
	
	

	
	//pt3x -= u2 = lam6^2-lam7*lam3^2
	dh_mybig_modsub_64(pt3->x,u2,pt3->x);

	

	//u3 = 2pt3x
	dh_mybig_moddouble_64(pt3->x,Pa,u3);
	//u2-=u3=lam7lam3^2-2pt3x=lam9
	dh_mybig_modsub_64(u2,u3,u2);

	

	//u1 *= pt3y = lam8*lam3^3
	dh_mybig_monmult_64(u1,pt3->y,u1);

	//pt3y = u2*u4 = lam9*lam6
	dh_mybig_monmult_64(u2,u4,pt3->y);

	// dh_mybig_copy(pt1->x,u1);
	// // dh_mybig_copy(pt1->y,u2);
	// return;

	//pt3y-=u1
	dh_mybig_modsub_64(pt3->y,u1,pt3->y);

	

	//pt3y/=2
	dh_mybig_monmult_64(pt3->y,mon_inv_two,pt3->y);
	// dh_mybig_half_64(pt3->y,pt3->y);

}

__device__ __host__ void dh_ellipticAdd_JJ_verbose(Jpoint *pt1, Jpoint* pt2,Jpoint* pt3){
	UINT64 u1[4],u2[4],u3[4],u4[4];
	//u1=z2^2

	#ifdef __CUDA_ARCH__
		const UINT64 *Pa=dc_p;
		const UINT64 *mon_inv_two = dc_mon_inv_two;

	#else
		const UINT64 *Pa=h_p;
		const UINT64 *mon_inv_two = h_mon_inv_two;

	#endif	
	printf("verbose mode-init: input pt1=%llx, pt2=%llx\n", pt1->x[0], pt2->x[0]);

	dh_mybig_monmult_64(pt2->z,pt2->z,u1);
	printf("verbose mode-u1: value=%llx\n", u1[0]);

	//u2=z1^2
	dh_mybig_monmult_64(pt1->z,pt1->z,u2);
	printf("verbose mode-u2: value=%llx\n", u2[0]);

	//u3 = u1*x1=x1*z2^2=lam1
	dh_mybig_monmult_64(pt1->x,u1,u3);
	printf("verbose mode-u3: value=%llx\n", u3[0]);	
	
	//u4 = u2*x2 = x2*z1^2=lam2
	dh_mybig_monmult_64(pt2->x,u2,u4);
	printf("verbose mode-u4: value=%llx\n", u4[0]);
	

	//u1 = z2^3
	dh_mybig_monmult_64(u1,pt2->z,u1);
	printf("verbose mode-u1: value=%llx\n", u1[0]);
	//u2 = z1^3
	dh_mybig_monmult_64(u2,pt1->z,u2);
	printf("verbose mode-u2: value=%llx\n", u2[0]);
	
	//u1 = u1*y1 = lam4
	dh_mybig_monmult_64(u1,pt1->y,u1);
	printf("verbose mode-lam4: value=%llx\n", u1[0]);
	//u2 = u2*y2 = lam5
	dh_mybig_monmult_64(u2,pt2->y,u2);
	printf("verbose mode-lam5: value=%llx\n", u2[0]);


	//pt3y = u3-u4 = lam3
	dh_mybig_modsub_64(u3,u4,pt3->y);
	printf("verbose mode-lam3: value=%llx\n", pt3->y[0]);

	

	//u3+=u4 = lam1+lam2 = lam7
	dh_mybig_modadd_64(u3,u4,u3);
	printf("verbose mode-lam7: value=%llx\n", u3[0]);
	//U4=u1-u2=lam4-lam5=lam6
	dh_mybig_modsub_64(u1,u2,u4);
	printf("verbose mode-lam6: value=%llx\n", u4[0]);

	
	//u1+=u2 = lam4+lam5=lam8
	dh_mybig_modadd_64(u1,u2,u1);
	printf("verbose mode-lam8: value=%llx\n", u1[0]);

	

	//pt3z = z1*z2*lam3
	dh_mybig_monmult_64(pt1->z,pt2->z,pt3->z);
	printf("verbose mode-z1*z2: value=%llx\n", pt3->z[0]);
	dh_mybig_monmult_64(pt3->z,pt3->y,pt3->z);
	printf("verbose mode-z1*z2*lam3: value=%llx\n", pt3->z[0]);

	
	//pt3x = u4^2 = lam6^2
	dh_mybig_monmult_64(u4,u4,pt3->x);
	printf("verbose mode-lam6^2: value=%llx\n", pt3->x[0]);
	//u2 = pt3y^2 = lam3^2
	dh_mybig_monmult_64(pt3->y,pt3->y,u2);
	printf("verbose mode-lam3^2: value=%llx\n", u2[0]);

	

	//pt3y = pt3y*u2 = lam3^3
	dh_mybig_monmult_64(pt3->y,u2,pt3->y);
	printf("verbose mode-lam3^3: value=%llx\n", pt3->y[0]);

	//u2 = u2*u3 = lam7*lam3^2
	dh_mybig_monmult_64(u2,u3,u2);
	printf("verbose mode-lam7*lam3^2: value=%llx\n", u2[0]);
	

	
	//pt3x -= u2 = lam6^2-lam7*lam3^2
	dh_mybig_modsub_64(pt3->x,u2,pt3->x);
	printf("verbose mode-pt3x: value=%llx\n", pt3->x[0]);
	

	//u3 = 2pt3x
	dh_mybig_moddouble_64(pt3->x,Pa,u3);
	printf("verbose mode-2pt3x: value=%llx\n", u3[0]);
	//u2-=u3=lam7lam3^2-2pt3x=lam9
	dh_mybig_modsub_64(u2,u3,u2);
	printf("verbose mode-lam9: value=%llx\n", u2[0]);

	

	//u1 *= pt3y = lam8*lam3^3
	dh_mybig_monmult_64(u1,pt3->y,u1);
	printf("verbose mode-lam8*lam3^3: value=%llx\n", u1[0]);

	//pt3y = u2*u4 = lam9*lam6
	dh_mybig_monmult_64(u2,u4,pt3->y);
	printf("verbose mode-lam9*lam6: value=%llx\n", pt3->y[0]);

	// dh_mybig_copy(pt1->x,u1);
	// // dh_mybig_copy(pt1->y,u2);
	// return;

	//pt3y-=u1
	dh_mybig_modsub_64(pt3->y,u1,pt3->y);
	printf("verbose mode-pt3y: value=%llx\n", pt3->y[0]);
	

	//pt3y/=2
	dh_mybig_monmult_64(pt3->y,mon_inv_two,pt3->y);
	printf("verbose mode-pt3y: value=%llx\n", pt3->y[0]);
	// dh_mybig_half_64(pt3->y,pt3->y);

}
//?????pt1???????pt2, pt1+=pt2
__device__ __host__ void dh_ellipticSumEqual_AJ2(Jpoint *pt1, Apoint* pt2)//pt1,pt2????????????????????????��?
{
	UINT64 u1[4],u2[4];
	if(dh_iszero_A(pt2))	return;	
	if(dh_iszero_J(pt1))
	{
		dh_mybig_copy(pt1->x, pt2->x);
		dh_mybig_copy(pt1->y, pt2->y);
		//Z????mon_ONE?????????????p????????
		//0x1000003d1L,0x0L,0x0L,0x0L
		pt1->z[0]=0x1000003d1L;		pt1->z[1]=0x0L;		pt1->z[2]=0x0L;		pt1->z[3]=0x0L;
		// pt1->z[4]=0x0000000000000000L;		pt1->z[5]=0x0000000000000001L;		pt1->z[6]=0x0000000100000000L;		pt1->z[7]=0x0000000000000000L;				
		return;
	}	
	//??????????��??????????????????????
	
	
	//3.????u1=(pt1->z)^2.
	dh_mybig_monmult_64(pt1->z, pt1->z, u1);
	
	//4.????u2=(pt1->z)*u1.
	dh_mybig_monmult_64(pt1->z, u1, u2);
	
	//5.????u1=(pt2->x)*u1.
	dh_mybig_monmult_64(pt2->x, u1, u1);
	
	//6.????u2=(pt2->y)*u2.
	dh_mybig_monmult_64(pt2->y, u2, u2);
	
	//7.????u1=u1-pt1->x.		
	dh_mybig_modsub_64(u1, pt1->x,u1);
	
	//8.????u2=u2-pt1->y.	
	dh_mybig_modsub_64(u2,pt1->y,u2);
	
	///*	
	//9.?��???,???????????????.
	if(dh_mybig_iszero(u1))
	{
		if(dh_mybig_iszero(u2))
		{
			//GAO:????????????????????
			ppoint_double(pt1,pt1);//y???????????????????
			printf("here! use ppoint double!\n");
			
			return;
		}
		else//?????????????????????
		{
			dh_setzero_J(pt1);
			return ;
		}
	}
	//*/
	//10.pt1->z=pt1->z*u1.
	dh_mybig_monmult_64(pt1->z, u1, pt1->z);
	
	//11.????pt2->x=u1^2.
	dh_mybig_monmult_64(u1, u1, pt2->x);
	
	//12.????pt2->y=pt2->x*u1.
	dh_mybig_monmult_64(u1, pt2->x, pt2->y);
	
	//13.????pt2->x=pt1->x*pt2->x.
	dh_mybig_monmult_64(pt1->x, pt2->x, pt2->x);
	
	//14.????u1=2*pt2->x.
	dh_mybig_modadd_64(pt2->x,pt2->x,u1);
	
	//15.x1=u2^2.
	dh_mybig_monmult_64(u2, u2, pt1->x);
	
	//16.x1=pt2->x
	dh_mybig_modsub_64(pt1->x,u1,pt1->x);
	
	//17.x1=x1-pt2->y
	dh_mybig_modsub_64(pt1->x,pt2->y,pt1->x);
	
	//18.????pt2->x=pt2->x-x1.
	dh_mybig_modsub_64(pt2->x,pt1->x,pt2->x);
	
	//19.pt2->x=pt2->x*u2
	dh_mybig_monmult_64(pt2->x, u2, pt2->x);
	
	//20.pt2->y=pt2->y*y1
	dh_mybig_monmult_64(pt2->y, pt1->y, pt2->y);
	
	//21.y1=pt2->x-pt2->y
	dh_mybig_modsub_64(pt2->x,pt2->y,pt1->y);
}
__device__ __host__ void dh_ellipticSumEqual_AJ(Jpoint *pt1, Apoint* pt2)//pt1,pt2????????????????????????��?
{
	UINT64 u1[4],u2[4],u3[4];
	if(dh_iszero_A(pt2))	return;	
	if(dh_iszero_J(pt1))
	{
		dh_mybig_copy(pt1->x, pt2->x);
		dh_mybig_copy(pt1->y, pt2->y);
		//Z????mon_ONE?????????????p????????
		//0x1000003d1L,0x0L,0x0L,0x0L
		pt1->z[0]=0x1000003d1L;		pt1->z[1]=0x0L;		pt1->z[2]=0x0L;		pt1->z[3]=0x0L;
		// pt1->z[4]=0x0000000000000000L;		pt1->z[5]=0x0000000000000001L;		pt1->z[6]=0x0000000100000000L;		pt1->z[7]=0x0000000000000000L;				
		return;
	}	
	//??????????��??????????????????????
	
	
	//3.????u1=(pt1->z)^2.
	dh_mybig_monmult_64(pt1->z, pt1->z, u1);
	
	//4.????u2=(pt1->z)*u1.
	dh_mybig_monmult_64(pt1->z, u1, u2);
	
	//5.????u1=(pt2->x)*u1.
	dh_mybig_monmult_64(pt2->x, u1, u1);
	
	//6.????u2=(pt2->y)*u2.
	dh_mybig_monmult_64(pt2->y, u2, u2);
	
	//7.????u1=u1-pt1->x.		
	dh_mybig_modsub_64(u1, pt1->x,u1);
	
	//8.????u2=u2-pt1->y.	
	dh_mybig_modsub_64(u2,pt1->y,u2);
	
	///*	
	//9.?��???,???????????????.
	if(dh_mybig_iszero(u1))
	{
		if(dh_mybig_iszero(u2))
		{
			//GAO:????????????????????
			ppoint_double(pt1,pt1);//y???????????????????
			printf("here! use ppoint double!\n");
			
			return;
		}
		else//?????????????????????
		{
			dh_setzero_J(pt1);
			return ;
		}
	}
	//*/
	//10.pt1->z=pt1->z*u1.
	dh_mybig_monmult_64(pt1->z, u1, pt1->z);

	// //11.????pt2->x=u1^2.
	// dh_mybig_monmult_64(u1, u1, pt2->x);
	//11.u3 = u1^2
	dh_mybig_monmult_64(u1,u1,u3);

	
	// //12.????pt2->y=pt2->x*u1.
	// dh_mybig_monmult_64(u1, pt2->x, pt2->y);
	//12.u1 = u1*u3
	dh_mybig_monmult_64(u1,u3,u1);
	
	//13.????pt2->x=pt1->x*pt2->x.
	// dh_mybig_monmult_64(pt1->x, pt2->x, pt2->x);
	//13.u3 = u3*pt1->x
	dh_mybig_monmult_64(u3, pt1->x, u3);
	
	//14.????u1=2*pt2->x.
	// dh_mybig_modadd_64(pt2->x,pt2->x,u1);
	//14.x1 = u2^2
	dh_mybig_monmult_64(u2, u2, pt1->x);
	
	//15.x1=u2^2.
	// dh_mybig_monmult_64(u2, u2, pt1->x);
	//15.x1 = x1-u3
	dh_mybig_modsub_64(pt1->x,u3,pt1->x);

	
	//16.x1=pt2->x
	// dh_mybig_modsub_64(pt1->x,u1,pt1->x);
	//16.x1=x1-u3
	dh_mybig_modsub_64(pt1->x,u3,pt1->x);
	
	//17.x1=x1-pt2->y
	// dh_mybig_modsub_64(pt1->x,pt2->y,pt1->x);
	//17.x1 = x1-u1
	dh_mybig_modsub_64(pt1->x,u1,pt1->x);
	

	//18.????pt2->x=pt2->x-x1.
	// dh_mybig_modsub_64(pt2->x,pt1->x,pt2->x);
	//18.u3 = u3 - x1
	dh_mybig_modsub_64(u3,pt1->x,u3);

	
	//19.pt2->x=pt2->x*u2
	// dh_mybig_monmult_64(pt2->x, u2, pt2->x);
	//19.u3 = u3*u2
	dh_mybig_monmult_64(u3, u2, u3);
	
	//20.pt2->y=pt2->y*y1
	// dh_mybig_monmult_64(pt2->y, pt1->y, pt2->y);
	//20.u1 = u1 * pt1->y
	dh_mybig_monmult_64(u1, pt1->y, u1);

	
	//21.y1=pt2->x-pt2->y
	// dh_mybig_modsub_64(pt2->x,pt2->y,pt1->y);
	//21.y1=u3-u1
	dh_mybig_modsub_64(u3,u1,pt1->y);
}

__device__ __host__ void dh_point_mult_inplace(Jpoint* pt1,UINT64 *k){
	//gyy
	int i,j;
	Jpoint tp;
	Jpoint t2;
	//????????1??��??

	//testcode
	// dh_mybig_copy(tp.x,pt1->x);
	// dh_mybig_copy(tp.y,pt1->y);
	// dh_mybig_copy(tp.z,pt1->z);



	// ppoint_double(pt1,pt1);
	// dh_ellipticAdd_JJ(pt1,&tp,pt1);
	
	// ppoint_double(pt1,pt1);
	// dh_ellipticAdd_JJ(pt1,&tp,pt1);
	
	// dh_mybig_copy(pt1->x,tp.x);
	// dh_mybig_copy(pt1->y,tp.y);
	// dh_mybig_copy(pt1->z,tp.z);

	for(i=3;i>=0;i--){
		if(k[i]!=0)	break;
	}
	if(i<0){
		printf("pm_in:k==0!!!!!!!!!\n");
	}
	for(j=63;j>=0;j--){
		if(((k[i]>>j)&0x01)!=0) break;
	}
	
	dh_mybig_copy(tp.x,pt1->x);
	dh_mybig_copy(tp.y,pt1->y);
	dh_mybig_copy(tp.z,pt1->z);
	j--;
	// printf("j=%d,i=%d\n",j,i);
	for(;j>=0;j--){
			// printf("double\n");
		ppoint_double(pt1,pt1);
		if(((k[i]>>j)&0x01)==1){
				// printf("add\n");
			dh_ellipticAdd_JJ(pt1,&tp,pt1);
		}
	}
	i--;
	for(;i>=0;i--){
		for(j=63;j>=0;j--){
				// printf("double\n");
			ppoint_double(pt1,pt1);
			if(((k[i]>>j)&0x01)==1){
					// printf("add\n");
				dh_ellipticAdd_JJ(pt1,&tp,pt1);
			}
		}
	}
	// printf("copy\n");
	// dh_mybig_copy(pt1->x,tp.x);
	// dh_mybig_copy(pt1->y,tp.y);
	// dh_mybig_copy(pt1->z,tp.z);
}

__device__ __host__ void dh_point_mult_finalversion(Jpoint* pt1,UINT64 *k,Jpoint* pt2){
	//gyy
	int i,j;
	Jpoint tp;
	// Jpoint t2;
	
	//????????1??��??

	//testcode
	// dh_mybig_copy(tp.x,pt1->x);
	// dh_mybig_copy(tp.y,pt1->y);
	// dh_mybig_copy(tp.z,pt1->z);

	// ppoint_double(pt1,pt1);
	// dh_ellipticAdd_JJ(pt1,&tp,pt1);
	
	// ppoint_double(pt1,pt1);
	// dh_ellipticAdd_JJ(pt1,&tp,pt1);
	
	// dh_mybig_copy(pt1->x,tp.x);
	// dh_mybig_copy(pt1->y,tp.y);
	// dh_mybig_copy(pt1->z,tp.z);

	for(i=3;i>=0;i--){
		if(k[i]!=0)	break;
	}
	if(i<0){
		printf("pm:k==0!!!!!!!!!\n");
	}
	for(j=63;j>=0;j--){
		if(((k[i]>>j)&0x01)!=0) break;
	}
	
	dh_mybig_copy(tp.x,pt1->x);
	dh_mybig_copy(tp.y,pt1->y);
	dh_mybig_copy(tp.z,pt1->z);
	j--;

	if(pt1==pt2){
		
		for(;j>=0;j--){
				// printf("double\n");
			ppoint_double(pt1,pt1);
			if(((k[i]>>j)&0x01)==1){
					// printf("add\n");
				dh_ellipticAdd_JJ(pt1,&tp,pt1);
			}
		}
		i--;
		for(;i>=0;i--){
			for(j=63;j>=0;j--){
					// printf("double\n");
				ppoint_double(pt1,pt1);
				if(((k[i]>>j)&0x01)==1){
						// printf("add\n");
					dh_ellipticAdd_JJ(pt1,&tp,pt1);
				}
			}
		}
	}else{
		
		// h_print_pointJ(pt1);
		// h_print_pointJ(&tp);
		for(;j>=0;j--){
				// printf("double\n");
			ppoint_double(&tp,&tp);
			if(((k[i]>>j)&0x01)==1){
					// printf("add\n");
				dh_ellipticAdd_JJ(&tp,pt1,&tp);
			}
		}
		i--;
		for(;i>=0;i--){
			for(j=63;j>=0;j--){
					// printf("double\n");
				ppoint_double(&tp,&tp);
				if(((k[i]>>j)&0x01)==1){
						// printf("add\n");
					dh_ellipticAdd_JJ(&tp,pt1,&tp);
				}
			}
		}
		// h_print_pointJ(&tp);
		dh_mybig_copy(pt2->x,tp.x);
		dh_mybig_copy(pt2->y,tp.y);
		dh_mybig_copy(pt2->z,tp.z);
	}
	// printf("j=%d,i=%d\n",j,i);
	
	// printf("copy\n");
	// dh_mybig_copy(pt1->x,tp.x);
	// dh_mybig_copy(pt1->y,tp.y);
	// dh_mybig_copy(pt1->z,tp.z);
}

__device__ __host__ void dh_point_mult_uint32(Jpoint* pt1, int k,Jpoint* pt2){
	int i;
	Jpoint tp;
	for(i=31;i>=0;i--){
		if((k>>i)&0x01!=0) break;
	}
	dh_mybig_copy(tp.x,pt1->x);
	dh_mybig_copy(tp.y,pt1->y);
	dh_mybig_copy(tp.z,pt1->z);
	i--;

	if(pt1==pt2){
		for(;i>=0;i--){
			// printf("double\n");
			ppoint_double(pt1,pt1);
			if(((k>>i)&0x01)==1){
				// printf("add\n");
				dh_ellipticAdd_JJ(pt1,&tp,pt1);
			}
		}
	}else{
		for(;i>=0;i--){
			// printf("double\n");
			ppoint_double(&tp,&tp);
			if(((k>>i)&0x01)==1){
				// printf("add\n");
				dh_ellipticAdd_JJ(&tp,pt1,&tp);
			}
		}
		dh_mybig_copy(pt2->x,tp.x);
		dh_mybig_copy(pt2->y,tp.y);
		dh_mybig_copy(pt2->z,tp.z);
	}

}

__device__ void d_base_point_mul(Jpoint *res,UINT64 *k){
	int i,j;
	Jpoint t;
	for(i=0;i<4;i++){
		// printf("k[%d]=%llx\n",i,k[i]);
		if(k[i]!=0) break;
	}
	// printf("i=%d\n",i);
	if(i==4){
		printf("basepm:k==0!!!!!!!!!\n");
	}
	for(j=0;j<64;j++){
		if(((k[i]>>j)&0x01)!=0) break;
	}
	// printf("j=%d\n",j);
	dh_mybig_copy(t.x,basePointMulPCT[i*64+j].x);
	dh_mybig_copy(t.y,basePointMulPCT[i*64+j].y);
	dh_mybig_copy(t.z,dc_mon_ONE);
	j++;
	for(;j<64;j++){

		if(((k[i]>>j)&0x01)==1){
			// printf("i=%d,j=%d\n",i,j);
			dh_ellipticSumEqual_AJ(&t,&basePointMulPCT[i*64+j]);
		}
	}
	i++;
	for(;i<4;i++){
		for(j=0;j<64;j++){
			if(((k[i]>>j)&0x01)==1){

				// printf("i=%d,j=%d\n",i,j); 
				dh_ellipticSumEqual_AJ(&t,&basePointMulPCT[i*64+j]);
				

			}
		}
	}
	dh_mybig_copy(res->x,t.x);
	dh_mybig_copy(res->y,t.y);
	dh_mybig_copy(res->z,t.z);

	
}


__device__ __host__ void dh_point_mult_outofplace(Jpoint* pt1,UINT64 *k,Jpoint* pt2){
	//gyy
	int i,j;

	//????????1??��??

	//testcode
	dh_mybig_copy(pt2->x,pt1->x);
	dh_mybig_copy(pt2->y,pt1->y);
	dh_mybig_copy(pt2->z,pt1->z);



	for(i=3;i>=0;i--){
		if(k[i]!=0)	break;
	}
	if(i<0){
		printf("pm_out:k==0!!!!!!!!!\n");
	}
	for(j=63;j>=0;j--){
		if(((k[i]>>j)&0x01)!=0) break;
	}
	

	j--;
	// printf("j=%d,i=%d\n",j,i);
	for(;j>=0;j--){
			// printf("double\n");
		ppoint_double(pt2,pt2);
		if(((k[i]>>j)&0x01)==1){
				// printf("add\n");
			dh_ellipticAdd_JJ(pt2,pt1,pt2);
		}
	}
	i--;
	for(;i>=0;i--){
		for(j=63;j>=0;j--){
				// printf("double\n");
			ppoint_double(pt2,pt2);
			if(((k[i]>>j)&0x01)==1){
					// printf("add\n");
				dh_ellipticAdd_JJ(pt2,pt1,pt2);
			}
		}
	}
	// printf("copy\n");
	// dh_mybig_copy(pt1->x,tp.x);
	// dh_mybig_copy(pt1->y,tp.y);
	// dh_mybig_copy(pt1->z,tp.z);
}


//GAO:��???
//????????
__device__ void d_multi_inverse(UINT64 *x)
{
	int i,j;
	UINT64 invlj[PARAL*8];//????????��??????4KB????4096=64*256,??????256??512??????
	UINT64 lmd[8];
	
	dh_mybig_copy(invlj,x);//	for(i=0;i<8;i++) invlj[0][i]=x[0][i];
		
	for(i=1;i<PARAL;i++)
	{			    					
		dh_mybig_monmult_64(invlj+(i-1)*8,x+i*8,invlj+i*8);
	}
		
	dh_mybig_monmult_64(invlj+(PARAL-1)*8,dc_ONE,lmd);//z=Z mod P //????????????
	dh_mybig_moninv(lmd,lmd); //?????????????2^n??
		
	for(i=PARAL-1;i>0;i--)
	{
		dh_mybig_monmult_64(invlj+(i-1)*8,lmd,invlj+i*8);
		dh_mybig_monmult_64(x+i*8,lmd,lmd);
		dh_mybig_copy(x+i*8,invlj+i*8);
		//for(j=0;j<8;j++) x[i*8+j]=invlj[i*8+j];	
	}	
	dh_mybig_copy(x,lmd);
	//for(j=0;j<8;j++) x[j]=lmd[j];	
}
//GAO:��???
//???????��?????
__global__ void d_multi_normlize_J(Jpoint *A, int n)
{	
	//????n?????PARAL????????
	
	int i,j,k;
	UINT64 tmp[8];
	int mytid=threadIdx.x+blockDim.x*blockIdx.x;
	int threadnum=blockDim.x*gridDim.x;
	UINT64 z[8*PARAL];//???z????��????????
			
	for(i=mytid*PARAL;i<n;i+=threadnum*PARAL) //n????PARAL??????????��????��??????��?????????
	{
		for(j=0;j<PARAL;j++)
		{
			for(k=0;k<8;k++)
				z[j*8+k]=A[i+j].z[k];
		}
				
		d_multi_inverse(z);
		
		for(j=0;j<PARAL;j++)
		{
			dh_mybig_monmult_64(z+j*8,z+j*8, tmp);
			dh_mybig_monmult_64(A[i+j].x,tmp, A[i+j].x);		
			dh_mybig_monmult_64(tmp,z+j*8, tmp);
			dh_mybig_monmult_64(A[i+j].y,tmp, A[i+j].y);

			//z??????mon_ONE?????mon_ONE??????��?
			A[i+j].z[0]=0x0000000000000001L;	A[i+j].z[1]=0x0000000100000000L;	A[i+j].z[2]=0x0000000000000000L;	A[i+j].z[3]=0x0000000100000000L;
			A[i+j].z[4]=0x0000000000000000L;	A[i+j].z[5]=0x0000000000000001L;	A[i+j].z[6]=0x0000000100000000L;	A[i+j].z[7]=0x0000000000000000L;	
			
			d_mon2normal_J(A+i+j);
		}
	}
	
}

/////////////////GPU?????????????////////////////////////////////////////////////////

__device__ __host__ void dh_my_point_copy(Jpoint* from, Jpoint* to) {
    dh_mybig_copy(to->x, from->x);
    dh_mybig_copy(to->y, from->y);
	// my problem: why here is x? i changed it to z.
    dh_mybig_copy(to->z, from->z);
}

/**
 * run_DBC:
 * use n (represented by DBC) calculate res = n * point, the (*) calculator means scalar multiple.
 *
 * @param dbc double-base chain's pointer.
 * @param point jacobian-montgomery base point.
 * @param res jacobian-montogomery result point. res can't be point.
 */
// __device__ __host__ void run_DBC(DBCv2* dbc, Jpoint* point, Jpoint* res) {
// #ifdef __CUDA_ARCH__
//     const UINT64 *Pa=dc_p;
// #else
//     const UINT64 *Pa=h_p;
// #endif

//     Jpoint base, wtf;
//     dh_my_point_copy(point, &base);
//     //dh_my_point_copy(point, &wtf);
//     int now_dbl = 0, now_tpl = 0;
//     if (dbc && dbc->length) {
//         for (int i = 0; i < dbc->length; i++) {
//             while (1) {
//                 if (now_dbl < dbc->data[i].dbl) {
//                     ppoint_double(&base, &wtf);
//                     dh_my_point_copy(&wtf, &base);
//                     now_dbl++;
//                 } else if (now_tpl < dbc->data[i].tpl) {
//                     ppoint_triple(&base, &wtf);
//                     dh_my_point_copy(&wtf, &base);
//                     now_tpl++;
//                 } else
//                     break;
//             } //here wtf = base = new dbl value.

//             if (dbc->data[i].minus) {
//                 // this actually goes for P - a*R, not P - a. but in montgomery field,
//                 // we actually want (P - a) * R, while (P - a)*R === P - a*R mod P.
//                 dh_mybig_sub_64(Pa, base.y, wtf.y);
//             }
//             if (i) { // i > 0
//                 dh_ellipticAdd_JJ(res, &wtf, res);
//             } else { // i == 0, just copy the first value.
//                 dh_my_point_copy(res, &wtf);
//             }
//         }
//     } else {
//         // dbc is null or dbc.length == 0. Now just do nothing.
//     }
// }

/**
 * run_DBC_v2:
 * use n (represented by DBC) calculate res = n * point, the (*) calculator means scalar multiple.
 *
 * @param dbc double-base chain's pointer.
 * @param point jacobian-montgomery base point.
 * @param res jacobian-montogomery result point. res can't be point.
 */
//#define DEBUG
__device__ int run_DBC_v2(Jpoint* pt1, Jpoint* res, int*DBC, int len) {
	// DBC is a int[len][3], DBC[i] = {stmp, alpha, beta}
#ifdef __CUDA_ARCH__
    const UINT64 *Pa=dc_p;
#else
    const UINT64 *Pa=h_p;
#endif
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int now_dbl = 0, now_tpl = 0;
	bool first = true;
	Jpoint base, tmp;
	dh_my_point_copy(pt1, &base);
	int cnt = 0;
	__syncthreads();
	// printf("tx=%d get point 0x%llx\n", threadIdx.x, pt1->x[0]);
	// printf("tx=%d gets base 0x%llx\n", threadIdx.x, base.x[0]);
	for (int i = len - 1; i >= 0; i--)
	{
		if (!i && bx == 13 && tx < 10 && tx % 2 == 1) {
		 	printf("bx=%d, tx=%d: DBC is %d %d\n", blockIdx.x, threadIdx.x, DBC[1], DBC[2]);
		 	__syncthreads();
		}
		while (1)
		{
			if (now_dbl < DBC[i*3 + 1])
			{
				ppoint_double(pt1, &tmp);
				if (now_dbl == 0) {
					//printf("tx=%d go to 2^0 3^0: point is %llx\n", tx, tmp.x[0]);
				}
				dh_my_point_copy(&tmp, pt1);
				now_dbl++;
			}
			else if (now_tpl < DBC[i*3 + 2])
			{
				// we dont have a good triple version.
				ppoint_double(pt1, &tmp);
				dh_ellipticAdd_JJ(pt1, &tmp, pt1);
				//ppoint_triple_v2(pt1, &tmp);
				//dh_my_point_copy(&tmp, pt1);
				//EC_POINT_tpl(group, mult_points, mult_points, ctx);
				now_tpl++;
			}
			else break;
			// if (tx == 0) {
			// 	printf("tx=%d go to 2^%d 3^%d: base is %llx\n", tx, now_dbl, now_tpl, pt1->x[0]);
			// }
			cnt++;
		}
		dh_my_point_copy(pt1, &tmp);
		if (DBC[i * 3] == -1)
		{
			dh_mybig_sub_64(Pa, pt1->y, tmp.y);
		}

		if (first)
		{
			dh_my_point_copy(&tmp, res);
			// if (threadIdx.x < 10)
			// 	printf("tx=%d first copy: res=%llx, from place %llx\n", threadIdx.x, res->x[0], tmp.x);
			first = false;
		} else {
			if (res->x[0] == tmp.x[0]) {ppoint_double(&tmp, res);} else
			//if (tx == 3 && bx == 0 && i == 59) { dh_ellipticAdd_JJ_verbose(res, &tmp, res);} else 
			dh_ellipticAdd_JJ(res, &tmp, res);
			//EC_POINT_add(group, r, r, mult_points, ctx);
		}
		if (tx == 3 && bx == 0) {
		 	//printf("go to item %d: res is %llx\n", i, res->x[0]);
		}
		cnt++;

	}
#ifdef DEBUG
	if (blockIdx.x == 0 && threadIdx.x < 10) {
		printf("tx=%d run %d cnts, returns 0x%llx\n",  threadIdx.x, cnt, res->x[0]);
	}
#endif
	return cnt;
}

#define MAX_BITS 256
__device__ void scalar_multiplication(UINT64* scalar, Jpoint* in, Jpoint* out) {
	int tx = threadIdx.x;
	bool first = true;
	Jpoint *tmp;
	tmp = (Jpoint*)malloc(sizeof(Jpoint));
	for (int i = 0; i < MAX_BITS / 64; i++) {
		for (int j = 0; j < 64; j++) {
			if (i + j == 0) {
				dh_mybig_copy(out->x, in->x);
    			dh_mybig_copy(out->y, in->y);
   		 		dh_mybig_copy(out->z, in->z);
				//if (tx == 0) printf("bit at %d: res is %llx\n", i * 64 + j, out->x[0]);
			} else if (scalar[i] & 1) {
				dh_ellipticAdd_JJ(out, tmp, out);
				//if (tx == 0) printf("bit at %d: res is %llx\n", i * 64 + j, out->x[0]);
			}
			scalar[i] /= 2;
			ppoint_double(in, tmp);
			dh_my_point_copy(tmp, in);
		}
	}
}
/**
 * get_DBC:
 * use n (represented by DBC) calculate res = n * point, the (*) calculator means scalar multiple.
 *
 * @param dbc double-base chain's pointer.
 * @param point jacobian-montgomery base point.
 * @param res jacobian-montogomery result point. res can't be point.
 */
#define DBC_store(x, y, z) *(DBC_store + (x) * DBC_level1 + (y) * DBC_level2 + (z))
__device__ int get_DBC(uint288* n, int*DBC_store, int* DBC_len) {
	int tx = threadIdx.x;
    int bx = blockIdx.x;
    int nthread = blockDim.x;
    int dbc_id = nthread * bx + tx;
	uint288 B;
	uint288 six_n;
	uint288 record_outer;
	uint288 temp_outer;
	uint64 n0;
	//int DBC_store[2][DBC_MAXLENGTH][3] = {0}; //��1ά����ͬ��DBC����2ά��һ��DBC�Ĳ�ͬ���3ά�����ţ�2�Ĵ�����3�Ĵ���
	//int DBC_len[2] = {0};
	const int DBC_level1 = 3 * DBC_MAXLENGTH;
	const int DBC_level2 = 3;
	
	//int bBound[MAX_2] = {0};

	//��nתΪ˫��������
	double dbl_n = n->to_double();
	//����B1,B2
	double B1 = 0.9091372900969896 * dbl_n; // 9*n/(7*sqrt(2))
	double B2 = 1.0774960475223583 * dbl_n; // 16*sqrt(2)*n/21
	//����LBound,RBound
	int LBound[MAX_3];
	int RBound[MAX_3];
	int DBC_index = 0;
	DBC_len[1] = 1 << 20; //��ʼ��Ϊ�㹻���������֤��ֻ�����1��DBCʱ��DBC��ʶ��ΪΪ��̵��Ǹ������������ж�
	for (int z = 0; z < DBC_COEF; z++)
	{
		int b = b_try[z];
		LBound[b] = log2(B1 / d_pow23[0][b]) + 1;
		RBound[b] = log2(B2 / d_pow23[0][b]);
		if (LBound[b] == RBound[b])
		{
			int a = RBound[b];
			int i = 0;
			int b_temp = b;
			uint288 t;
			// OCCULTICPLUS: for many reason we need to cancel the copy function for uint288, so explicit deep copy.
			for (int i = 0; i < 9; i++) {
				t.data[i] = n->data[i];
			}
			int s = 1;
			while (!t.iszero())
			{
				//����alpha,beta
				double dbl_t = t.to_double();
				int alpha = a, beta = b_temp;
				double logt = log2(dbl_t);
				const double log3 = log2(3.0);
				for (int j = b_temp; j >= max(0, b_temp - 6); j--)
				{
					int alpha_j;
					if (d_pow23[0][j] >= dbl_t) {
						alpha_j = 0;
					} else {
						int k_j = int(logt - j * log3);
						if (k_j >= a)
							alpha_j = a;
						else
						{
							if (abs(dbl_t - d_pow23[k_j][j]) <= abs(d_pow23[k_j + 1][j] - dbl_t))
								alpha_j = k_j;
							else
								alpha_j = k_j + 1;
						}
					}
					if (abs(dbl_t - d_pow23[alpha_j][j]) <= abs(d_pow23[alpha][beta] - dbl_t))
					{
						alpha = alpha_j;
						beta = j;
					}
				}
				int stmp = s;
				if (!(t >= u_pow23[alpha][beta]))
				 	s = -s;
				DBC_store(DBC_index, i, 0) = stmp;
				DBC_store(DBC_index, i, 1) = alpha;
				DBC_store(DBC_index, i, 2) = beta;
				// in case t not be 0.
				//if (i % 10 == 0) t.data[i / 10] = 0;
				i++;
				if (t >= u_pow23[alpha][beta])
				 	t = t - u_pow23[alpha][beta];
				 else
				 	t = u_pow23[alpha][beta] - t;
				a = alpha;
				b_temp = beta;
	#ifdef DEBUG
				if (tx == 1 && i % 10 == 0) {
					printf("check tx1 data, iteration=%d: ", i);
					for (int i = 0; i < 9; i++) {
						printf("0x%x ", t.data[i]);
					}
					printf("\ncheckminus data %d %d:\n", alpha, beta);
					for (int i = 0; i < 9; i++) {
						printf("0x%x ", u_pow23[alpha][beta].data[i]);
					}
					printf("\n");
				}
				if (i > 100) {
					printf("tx = %d exception: infinite loop!");
					for (int i = 0; i < 9; i++) {
						printf("0x%x ", t.data[i]);
					}
					printf("\ncheckminus data %d %d:\n", alpha, beta);
					for (int i = 0; i < 9; i++) {
						printf("0x%x ", u_pow23[alpha][beta].data[i]);
					}
				}
	#endif
			}
			DBC_len[DBC_index] = i;
			int temp0 = DBC_len[0] * ADD_COST + DBC_store(0, 0, 1) * DBL_COST + DBC_store(0, 0, 2) * TPL_COST;
			int temp1 = DBC_len[1] * ADD_COST + DBC_store(1, 0, 1) * DBL_COST + DBC_store(1, 0, 2) * TPL_COST;
			if (temp0 < temp1)
				DBC_index = 1;
			else
				DBC_index = 0;
		}
	}
	int min_index = 1 - DBC_index;
	return min_index;
}

/////////////////occultic plus's work: use DBC to calculate ended.////////////////////////////////////////////////////
__global__ void d_get_para(UINT64 *para)
{
	int i;
	UINT64 *pt64;
	
	if(threadIdx.x==0 && blockIdx.x==blockDim.x-1)
	{
		pt64=para+  0; for(i=0;i<8;i++) pt64[i]=dc_p[i];
		pt64=para+ 56; for(i=0;i<8;i++) pt64[i]=dc_ONE[i];
		pt64=para+ 64; for(i=0;i<8;i++) pt64[i]=dc_mon_ONE[i];						
	}	
}

void h_print_para()
{
	int groups=11;
	UINT64 *testdata=(UINT64 *)malloc(groups*8*8);	
	UINT64 *d_testdata;
	
	HANDLE_ERROR( cudaMalloc((void**)&d_testdata, groups*8*8) );
	
	d_get_para<<<BLOCKNUM,BLOCKSIZE>>>(d_testdata);
	
	HANDLE_ERROR( cudaMemcpy( testdata, d_testdata,	64*groups, cudaMemcpyDeviceToHost));
	printf("\n__const__ UINT64 testdc_p[8]=");h_mybig_print(testdata);
	printf("\n__const__ UINT64 testdc_ONE[8]=");h_mybig_print(testdata+56);
	printf("\n__const__ UINT64 testdc_mon_ONE[8]=");h_mybig_print(testdata+64);
	
	free(testdata);
	HANDLE_ERROR(cudaFree(d_testdata));
}



int h_get_gpu_info()
{
	  cudaDeviceProp  prop;

    int count;
    HANDLE_ERROR( cudaGetDeviceCount( &count ) );
    
    for (int i=0; i< count; i++) {
        HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
        printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop.name );
        printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
        printf( "Clock rate:  %d\n", prop.clockRate );
        printf( "Device copy overlap:  " );
        if (prop.deviceOverlap)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n");
        printf( "Kernel execution timeout :  " );
        if (prop.kernelExecTimeoutEnabled)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );

        printf( "   --- Memory Information for device %d ---\n", i );
        printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
        printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
        printf( "Max mem pitch:  %ld\n", prop.memPitch );
        printf( "Texture Alignment:  %ld\n", prop.textureAlignment );

        printf( "   --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count:  %d\n",
                    prop.multiProcessorCount );
        printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per mp:  %d\n", prop.regsPerBlock );
        printf( "Threads in warp:  %d\n", prop.warpSize );
        printf( "Max threads per block:  %d\n",
                    prop.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n",
                    prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                    prop.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",
                    prop.maxGridSize[0], prop.maxGridSize[1],
                    prop.maxGridSize[2] );
        printf( "\n" );
    }

	return 1;
}
