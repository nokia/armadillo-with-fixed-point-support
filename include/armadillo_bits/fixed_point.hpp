/**********************************************************************************
 *                          Fixed point class
 **********************************************************************************/

/** Check supported types */
template<typename T> struct is_supported_type      {const static bool value = false;};
template<>           struct is_supported_type<s8>  {const static bool value = true;};
template<>           struct is_supported_type<u8>  {const static bool value = true;};
template<>           struct is_supported_type<s16> {const static bool value = true;};
template<>           struct is_supported_type<u16> {const static bool value = true;};
template<>           struct is_supported_type<s32> {const static bool value = true;};
template<>           struct is_supported_type<u32> {const static bool value = true;};

#if (defined(ARMA_USE_U64S64) || defined(ARMA_USE_U64S64_FIXED_POINT))
template<>           struct is_supported_type<s64> {const static bool value = true;};
template<>           struct is_supported_type<u64> {const static bool value = true;};
#endif

#if defined(ARMA_USE_U128S128)
template<>           struct is_supported_type<s128>{const static bool value = true;};
template<>           struct is_supported_type<u128>{const static bool value = true;};
#endif

/** Multiplication & division cast type */
template<typename T> struct mult_div_cast_type       {                  };
template<>           struct mult_div_cast_type <s8>  {typedef s16  type;};
template<>           struct mult_div_cast_type <u8>  {typedef u16  type;};
template<>           struct mult_div_cast_type <s16> {typedef s32  type;};
template<>           struct mult_div_cast_type <u16> {typedef u32  type;};

#if (defined(ARMA_USE_U64S64) || defined(ARMA_USE_U64S64_FIXED_POINT))
template<>           struct mult_div_cast_type <s32> {typedef s64   type;};
template<>           struct mult_div_cast_type <u32> {typedef u64   type;};
#endif

#if (defined(ARMA_USE_U64S64) || defined(ARMA_USE_U64S64_FIXED_POINT)) && defined(ARMA_USE_U128S128)
template<>           struct mult_div_cast_type <s64> {typedef s128  type;};
template<>           struct mult_div_cast_type <u64> {typedef u128  type;};
#endif

#if defined(ARMA_USE_U128S128)
template<>           struct mult_div_cast_type <s128>{typedef s128  type;};
template<>           struct mult_div_cast_type <u128>{typedef u128  type;};
#endif

/** Approve base type */
template<bool, typename type> struct approve_type_if            {                    };
template<      typename type> struct approve_type_if<true, type>{typedef type result;};

template<typename BT, u16 P> /** P(precision)*/
class FP
{
public:
    typedef typename approve_type_if<(is_supported_type<BT>::value) && (P < 8 * sizeof(BT)), BT>::result T; /** typdef of base type*/
    typedef FP<T, P>                              elem_type;
    typedef typename mult_div_cast_type<T>::type  T1;
    static const T FIXED_ONE = static_cast<BT>(1) << P;

private:
    T v; /** the fixed point value */

public:
    /** define class as friend to itself for
    every precision PP */
    template<u16 PP>
    friend class FP<BT, P>;

    /** Default constructor */
    FP():v(0){}

    /** Copy constructor */
    FP(const FP& x):v(x.v){}
    inline const T getIntValue() const{return v;}
    inline void setIntValue(T val) {v = val;}
    /** Create from another fp precision and different or same base type */
    template<typename BTin, u16 PP>
    FP(const FP<BTin, PP>& x){
        u16 diff_precision = std::abs(PP-P);
        if (PP > P){
            v = static_cast<BT>(typename mult_div_cast_type<BTin>::type(x.v) >> diff_precision);
        }
        else {
            v = static_cast<BT>(typename mult_div_cast_type<BTin>::type(x.v) << diff_precision);
        }
    }

    /** Create from int */
    FP(const s32& x):v(x * FIXED_ONE){
    }

    /** Create from unsigned int */
    FP(const u32& x):v(x * FIXED_ONE){
    }

    /** Create from long int */
    FP(const slng_t& x):v(x * FIXED_ONE){
    }

    /** Create from unsigned long int */
    FP(const ulng_t& x):v(x * FIXED_ONE){
    }

    /** Create from float */
    FP(const float& x):v(x * FIXED_ONE){
    }

    /** Create from double */
    FP(const double& x):v(x * FIXED_ONE){
    }

#if (defined(ARMA_USE_U64S64) || defined(ARMA_USE_U64S64_FIXED_POINT))
    /** Create from LL */
    FP(const s64& x):v(x * FIXED_ONE){
    }

    /** Create from ULL */
    FP(const u64 & x):v(x * FIXED_ONE){
    }
#endif

#if defined(ARMA_USE_U128S128)
    /** Create from 128 bit integer */
    FP(const s128& x):v(x * FIXED_ONE){
    }

    /** Create from U128 integer */
    FP(const u128& x):v(x * FIXED_ONE){
    }
#endif

    /** implicit cast to double */
    operator double () const {return static_cast<double>(v)/FIXED_ONE;}

    /** explicit cast to other fundamental arithmetic types */
    template<typename T>
    T castTo()const{return static_cast<T>(v)/FIXED_ONE;}

    /** Cast to other FP and other base type */
    template<typename BTout, u16 PP>
    inline operator FP<BTout, PP>() const {
        return(FP<BTout, PP>(this));
    }

    /** Assignment of FP different precision and different base type */
    template<typename BTin, u16 PP>
    inline const FP operator=(const FP<BTin, PP>& x){
        *this = FP(x);
        return *this;
    }

    /** Assignment of FP */
    inline const FP operator=(const FP& x){
        v = x.v;
        return *this;
    }

    /** Assignment of int32 */
    inline const FP operator=(const s32& x){
        return (*this = FP(x));
    }

    /** Assignment of unsigned int32  */
    inline const FP operator=(const u32& x){
        return (*this = FP(x));
    }

#if (defined(ARMA_USE_U64S64) || defined(ARMA_USE_U64S64_FIXED_POINT))
    /** Assignment of int64 */
    inline const FP operator=(const s64& x){
        return (*this = FP(x));
    }

    /** Assignment of unsigned int64  */
    inline const FP operator=(const u64& x){
        return (*this = FP(x));
    }
#endif

#if defined(ARMA_USE_U128S128)
    /** Assignment of int128 */
    inline const FP operator=(const s128& x){
        return (*this = FP(x));
    }

    /** Assignment of unsigned int128  */
    inline const FP operator=(const u128& x){
        return (*this = FP(x));
    }
#endif

    /** Assignment of float */
    inline const FP operator=(const float& x){
        return (*this = FP(x));
    }

    /** Assignment of double */
    inline const FP operator=(const double& x){
        return (*this = FP(x));
    }

    /** Addition assignment */
    inline const FP& operator+=(const FP& x){
        v += x.v;
        return *this;
    }

    /** Addition */
    inline const FP operator+(const FP& x) const{
        FP res(*this);
        return res+=x;
    }

    /** Addition assignment with other types */
    template<typename TT>
    inline const FP& operator+=(const TT& x){
        return *this+=FP(x);
    }

    /** Addition with other types */
    template<typename TT>
    inline const FP operator+(const TT& x) const{
        FP res(*this);
        return res+=FP(x);
    }

    /** Minus */
    inline const FP operator-(){
        return *this *= -1;
    }

    /** Subtraction assignment */
    inline const FP& operator-=(const FP& x){
        v -= x.v;
        return *this;
    }

    /** Subtraction */
    inline const FP operator-(const FP& x) const{
        FP res(*this);
        return res-=x;
    }

    /** Subtraction assignment with other types */
    template<typename TT>
    inline const FP &operator-=(const TT& x){
        return *this-=FP(x);
    }

    /** Subtraction with other types */
    template<typename TT>
    inline const FP operator-(const TT& x)const{
        FP res(*this);
        return res-=FP(x);
    }

    /** Multiplication assignment */
    inline const FP& operator *=(const FP& x) {
        v = static_cast<T>((static_cast<T1>(v) * x.v) >> P);
        return *this;
    }

    /** Multiplication */
    inline const FP operator *(const FP& x) const{
        FP res(*this);
        return res*=x;
    }

    /** Multiplication with other precision types*/
    template<u16 PP>
    inline const FP operator *(const FP<BT, PP>& x) const{
        FP res(*this);
        return res*=FP(x);
    }

    inline const FP operator *(const float& x) const {
        FP res(*this);
        return res*=FP(x);
    }

    inline const FP operator *(const double& x) const {
        FP res(*this);
        return res*=FP(x);
    }

    inline const FP operator *(const s32& x) const {
        FP res(*this);
        return res*=FP(x);
    }

    inline const FP operator *(const u32& x) const {
        FP res(*this);
        return res*=FP(x);
    }

#if (defined(ARMA_USE_U64S64) || defined(ARMA_USE_U64S64_FIXED_POINT))
    inline const FP operator *(const s64& x) const {
        FP res(*this);
        return res*=FP(x);
    }

    inline const FP operator *(const u64& x) const {
        FP res(*this);
        return res*=FP(x);
    }
#endif

#if defined(ARMA_USE_U128S128)
    inline const FP operator *(const s128& x) const {
        FP res(*this);
        return res*=FP(x);
    }

    inline const FP operator *(const u128& x) const {
        FP res(*this);
        return res*=FP(x);
    }
#endif

    /** Division assignment */
    inline const FP& operator /=(const FP& x) {
        v = static_cast<T>((static_cast<T1>(v) << P)/x.v);
        return *this;
    }

    /** Division */
    inline const FP operator/(const FP& x) const {
        FP res(*this);
        return res/=x;
    }

    /** Division by other types*/
    template<typename T>
    inline const FP operator/(const T& x) const {
        FP res(*this);
        return res/=FP(x);
    }

    /** Bitwise shift up assignment */
    inline const FP& operator<<=(const u16 x){
        v <<= x;
        return * this;
    }

    /** Bitwise shift up */
    inline const FP operator<<(const u16 x) const{
        FP res(*this);
        return res<<=x;
    }

    /** Smaller than */
    inline bool operator<(const FP& x) const{
        return (v < x.v);
    }

    /** Smaller than with other type */
    template<typename TT>
    inline bool operator<(const TT& x) const{
        return (*this < FP(x));
    }

    /** Greater than */
    inline bool operator>(const FP& x) const{
        return (x < *this);
    }

    /** Greater than with other type */
    template<typename TT>
    inline bool operator>(const TT& x) const{
        return (*this > FP(x));
    }

    /** Smaller than or equal */
    inline bool operator<=(const FP& x) const{
        return !(*this > x);
    }

    /** Smaller than or equal with other type */
    template<typename TT>
    inline bool operator<=(const TT& x) const{
        return (*this <= FP(x));
    }

    /** Greater than or equal */
    inline bool operator>=(const FP& x) const{
        return !(*this < x);
    }

    /** Greater than or equal with other type */
    template<typename TT>
    inline bool operator>=(const TT& x) const{
        return (*this >= FP(x));
    }

    /** Equal operator*/
    inline bool operator==(const FP& x) const{
        return (v == x.v);
    }

    /** Equal with other type */
    template<typename TT>
    inline bool operator==(const TT& x) const{
        return (*this == FP(x));
    }

    /** Inequality operator*/
    bool operator!=(const FP& x) const{
        return !(*this == x);
    }

    /** Inequality operator with other type*/
    template<typename TT>
    inline bool operator!=(const TT& x) const{
        return !(*this == FP(x));
    }
};

/** Left hand side multiplication with other types */
template<typename BT, typename LHSTYPE, u16 P>
inline const FP<BT, P> operator*(LHSTYPE const& lhs, const FP<BT, P>&fp){
    FP<BT, P> res(fp);
    return res*= FP<BT, P>(lhs);
};

/** Specialization for multiplication of complex numbers.
 *  Avoid ambiguity with * operator in std::complex. 
*/
template<typename BT, u16 P>
inline const std::complex<FP<BT, P> > operator*(std::complex<FP<BT, P> > const& lhs, const FP<BT, P>&fp){
    std::complex<FP<BT, P> > res(fp);
    return res *= lhs;
};

/** Left hand side multiplication specialization for U64S64 */
#if (defined(ARMA_USE_U64S64) || defined(ARMA_USE_U64S64_FIXED_POINT))
template<typename BT, u16 P>
inline const FP<BT, P> operator*(s64 const& lhs, const FP<BT, P>&fp){
    FP<BT, P> res(fp);
    return res*= FP<BT, P>(lhs);
};

template<typename BT, u16 P>
inline const FP<BT, P> operator*(u64 const& lhs, const FP<BT, P>&fp){
    FP<BT, P> res(fp);
    return res*= FP<BT, P>(lhs);
};
#endif

/** Left hand side multiplication specialization for U128S128 */
#if defined(ARMA_USE_U128S128)
template<typename BT, u16 P>
inline const FP<BT, P> operator*(s128 const& lhs, const FP<BT, P>&fp){
    FP<BT, P> res(fp);
    return res*= FP<BT, P>(lhs);
};

template<typename BT, u16 P>
inline const FP<BT, P> operator*(u128 const& lhs, const FP<BT, P>&fp){
    FP<BT, P> res(fp);
    return res*= FP<BT, P>(lhs);
};
#endif

/** Left hand side multiplication with other fixed point using different precision */
template<typename BT, u16 P, u16 PP>
inline FP<BT, P> operator*(const FP<BT, PP>& lhs, const FP<BT, P>&fp){
    FP<BT, P> res(lhs);
    return res*= fp;
};

//! @}
