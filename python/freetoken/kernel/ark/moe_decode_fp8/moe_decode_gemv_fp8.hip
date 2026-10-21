#include "ft_decode_impl.h"
extern "C" int radeon_moe_decode_gemv_fp8(const void* x,const void* w1,const void* s1,const void* w2,const void* s2,
 const void* ids,const void* tw,void* h,void* out,int T,int K,int H,int I,int ss1,int ss2,void* stream){
 ark_ft::launch<true>(x,w1,s1,w2,s2,ids,tw,h,out,T,K,H,I,ss1,ss2,stream);
 return (int)hipGetLastError();
}
