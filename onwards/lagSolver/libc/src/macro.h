#ifndef _MACRO_H_ 
#define _MACRO_H_

#define PI 3.14159265359

#define ALLOC(type)    ((type*) malloc(sizeof(type)))
#define ALLOCN(type,n) ((type*) malloc(sizeof(type))*n)

#define VECINT(a_) (int*)    calloc( a_, sizeof(int) )
#define VEC(a_)    (double*) calloc( a_, sizeof(double) )

#define NORM(a_,b_) sqrt( a_*a_ + b_*b_ )
#define DOT(a_,b_)  (a_[0]*b_[0] + a_[1]*b_[1])

#define POW2WSIGN(a_) copysign(a_*a_,a_)
#define SQRTWSIGN(a_) copysign(sqrt(fabs(a_)),a_)

#define I2IP(a_,b_) (b_-a_->i0+a_->n)%a_->n
#define IP2I(a_,b_) (b_+a_->i0+a_->n)%a_->n

#define FREE(type_, ptr_) free_##type_(ptr_) ; free(ptr_)

#define COPYPORDER(strct_,in_,out_,comp_) \
    int i_; \
    for (i_ = 0; i_ < strct_->n; i_++) { \
        out_[i_] = strct_->in_[IP2I(strct_,i_)][comp_]; \
    }

#define VECTORIZE(f_, strct_, xv_, yv_, n_tot_, zv_) \
    int i_; \
    double *xv_loc_ = VEC(2);\
    double *zv_loc_ = VEC(2);\
    for (i_ = 0; i_ < n_tot_; i_++) { \
        xv_loc_[0] = xv_[i_];  xv_loc_[1] = yv_[i_]; \
        f_(strct_, xv_loc_, zv_loc_); \
        zv_[i_] = zv_loc_[0]; zv_[i_+n_tot] = zv_loc_[1]; \
    } \
    free(xv_loc_);\
    free(zv_loc_);

#define VECTORIZEPORDER(f_, strct_, xv_, yv_, n_tot_, zv_) \
    int i_; \
    double *xv_loc_ = VEC(2);\
    for (i_ = 0; i_ < n_tot_; i_++) { \
        xv_loc_[0] = xv_[IP2I(strct_,i_)];  xv_loc_[1] = yv_[IP2I(strct_,i_)]; \
        zv_[i_] = f_(strct_, xv_loc_); \
    } \
    free(xv_loc_);

#define COPY(strct_,in_,out_,comp_) \
    int i_; \
    for(i_ = 0; i_ < strct_->n; i_++) { \
        out_[i_] = strct_->in_[i_][comp_] \
    }

#endif
