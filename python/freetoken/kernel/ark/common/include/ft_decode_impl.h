// ARK MoE decode with FreeToken intermediate rounding, without atomics or a memset.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
namespace ark_ft {
using bf16=__hip_bfloat16;
__device__ __forceinline__ float sum(float x){for(int s=16;s;s>>=1)x+=__shfl_xor(x,s);return x;}
__device__ __forceinline__ float e4m3(unsigned char x){
 #if defined(__gfx1201__)
 return __builtin_amdgcn_cvt_f32_fp8((int)x,0);
#else
 unsigned short u=((unsigned)(x&128)<<8)|((unsigned)(x&127)<<7);__half v;__builtin_memcpy(&v,&u,2);return __half2float(v)*256.f;
#endif
}
template<bool FP8> __device__ __forceinline__ float product(float a,float b){
 if constexpr(FP8)return a*b;else return (float)(bf16)(a*b);
}
template<bool FP8,int WARPS>
__global__ __launch_bounds__(WARPS*32) void gate_up(const bf16* x,const void* weights,const bf16* scales,
 const int* ids,const float* tw,float* h,int T,int K,int H,int I,int ss){
 int row=blockIdx.x*WARPS+(threadIdx.x>>5),lane=threadIdx.x&31;
 if(row>=T*K*I)return;
 int j=row%I,route=row/I,t=route/K,e=ids[route];
 if(e<0 || tw[route]==0.f){if(lane==0)h[row]=0.f;return;}
 float ag=0.f,au=0.f;
 for(int kb=0;kb<H/128;++kb){
  int k=kb*128+lane*4;float2 xr=*(const float2*)(x+(size_t)t*H+k);auto xx=(const bf16*)&xr;
  if constexpr(FP8){
   auto w=(const unsigned char*)weights;
   unsigned gw=*(const unsigned*)(w+((size_t)e*2*I+j)*H+k),uw=*(const unsigned*)(w+((size_t)e*2*I+I+j)*H+k);
   float pg=0.f,pu=0.f;
   #pragma unroll
   for(int z=0;z<4;++z){pg+=e4m3((gw>>(8*z))&255)*(float)xx[z];pu+=e4m3((uw>>(8*z))&255)*(float)xx[z];}
   ag+=pg*(float)scales[((size_t)e*(2*I/128)+j/128)*ss+kb];au+=pu*(float)scales[((size_t)e*(2*I/128)+(I+j)/128)*ss+kb];
  }else{
   auto w=(const bf16*)weights;
   float2 gr=*(const float2*)(w+((size_t)e*2*I+j)*H+k),ur=*(const float2*)(w+((size_t)e*2*I+I+j)*H+k);
   auto gg=(const bf16*)&gr;auto uu=(const bf16*)&ur;
   #pragma unroll
   for(int z=0;z<4;++z){ag+=product<false>((float)gg[z],(float)xx[z]);au+=product<false>((float)uu[z],(float)xx[z]);}
  }
 }
 ag=(float)(bf16)sum(ag);au=(float)(bf16)sum(au);
 if(lane==0)h[row]=(float)(bf16)((ag/(1.f+expf(-ag)))*au);
}
template<bool FP8,int WARPS>
__global__ __launch_bounds__(WARPS*32) void down_reduce(const float* h,const void* weights,const bf16* scales,
 const int* ids,const float* tw,bf16* out,int T,int K,int H,int I,int ss){
 int row=blockIdx.x*WARPS+(threadIdx.x>>5),lane=threadIdx.x&31;if(row>=T*H)return;
 int t=row/H,n=row%H;float total=0.f;
 for(int route=0;route<K;++route){
  int off=t*K+route,e=ids[off];float weight=tw[off];if(e<0 || weight==0.f)continue;
  float acc=0.f;
  for(int kb=0;kb<I/128;++kb){
   int k=kb*128+lane*4;float4 hh=*(const float4*)(h+(size_t)off*I+k);auto hv=(const float*)&hh;
   if constexpr(FP8){
    auto w=(const unsigned char*)weights;unsigned v=*(const unsigned*)(w+((size_t)e*H+n)*I+k);float p=0.f;
    #pragma unroll
    for(int z=0;z<4;++z)p+=e4m3((v>>(8*z))&255)*hv[z];
    acc+=p*(float)scales[((size_t)e*(H/128)+n/128)*ss+kb];
   }else{
    auto w=(const bf16*)weights;float2 wr=*(const float2*)(w+((size_t)e*H+n)*I+k);auto wv=(const bf16*)&wr;
    #pragma unroll
    for(int z=0;z<4;++z)acc+=product<false>((float)wv[z],hv[z]);
   }
  }
  total+=(float)(bf16)(sum(acc)*weight);
 }
 if(lane==0)out[row]=(bf16)total;
}
template<bool FP8,int WARPS=4>
void launch(const void* x,const void* w1,const void* s1,const void* w2,const void* s2,const void* ids,const void* tw,
 void* h,void* out,int T,int K,int H,int I,int ss1,int ss2,void* stream){
 auto s=(hipStream_t)stream;
 gate_up<FP8,WARPS><<<dim3((T*K*I+WARPS-1)/WARPS),dim3(WARPS*32),0,s>>>((const bf16*)x,w1,(const bf16*)s1,(const int*)ids,(const float*)tw,(float*)h,T,K,H,I,ss1);
 down_reduce<FP8,WARPS><<<dim3((T*H+WARPS-1)/WARPS),dim3(WARPS*32),0,s>>>((const float*)h,w2,(const bf16*)s2,(const int*)ids,(const float*)tw,(bf16*)out,T,K,H,I,ss2);
}
}
