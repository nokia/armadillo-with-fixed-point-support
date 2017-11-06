//#define CATCH_CONFIG_MAIN

#include <armadillo>
#include "catch.hpp"

using namespace arma;

#define SINGLE_PRECISION 23u
#define DOUBLE_PRECISION 52u
#define ERROR_PRECISION  pow(10, -5)
#define ERROR_DPRECISION pow(10, -15)
#define NBROWS 16
#define NBCOLS 16

#if defined(ARMA_USE_U64S64)
typedef FP<s64, SINGLE_PRECISION> fixed64_23;
typedef std::complex<FP<s64, SINGLE_PRECISION> > cx_fixed64_23;
typedef Mat<fixed64_23> fixed_fmat;
typedef Mat<cx_fixed64_23> cx_fixed_fmat;
#endif

#if defined(ARMA_USE_U128S128)
typedef FP<s128, DOUBLE_PRECISION> fixed128_52;
typedef std::complex<FP<s128, DOUBLE_PRECISION> > cx_fixed128_52;
typedef Mat<fixed128_52> fixed_mat;
typedef Mat<cx_fixed128_52> cx_fixed_mat;
#endif


//******************************************************************************//
//*              Unit test related to fixed point class it'self                *//
//******************************************************************************//

/** Addition of two fixed point number*/ 
TEST_CASE("addition"){
    FP<s64, SINGLE_PRECISION> fp1 = 1.5f;
    FP<s64, SINGLE_PRECISION> fp2 = 2.5f;
	REQUIRE((fp1 + fp2) == (1.5f + 2.5f));
}


/** Addition with none fixed point number*/ 
TEST_CASE("additionWithNoneFixedPointNumber"){
    FP<s64, SINGLE_PRECISION> fp =12.556f;
	REQUIRE((5.45f + fp + 3.2578) == Approx(5.45f + 12.556f + 3.2578).epsilon(ERROR_PRECISION));
}

TEST_CASE("lhsAddition"){
    const float number = 5.45f;
    FP<s64, SINGLE_PRECISION> fp = 12.556f;
	REQUIRE((number + fp) == Approx(number + 12.556f).epsilon(ERROR_PRECISION));
}

TEST_CASE("rhsAddition"){
    const float number = 5.45f;
    FP<s64, SINGLE_PRECISION> fp = 12.556f;
	REQUIRE((float)(fp + number) == Approx(12.556f + number).epsilon(ERROR_PRECISION));
}


/** Mixed fixed point addition different precision*/ 
TEST_CASE("mixedFpAddition"){
    FP<s64, SINGLE_PRECISION> fp1 = 1.515863f; 
    FP<s32, 16> fp2 = 2.585f;
	REQUIRE((float)(fp1 + fp2) == Approx(1.515863f + 2.585f).epsilon(ERROR_PRECISION));
}


/** Mixed fixed point substraction different precision*/ 
TEST_CASE("mixedFpSubstraction"){
    FP<s64, SINGLE_PRECISION> fp1 = 1.515863f; 
    FP<s32, 16> fp2 = 2.585f;
	REQUIRE((float)(fp1 - fp2) == Approx(1.515863f - 2.585f).epsilon(ERROR_PRECISION));
}


TEST_CASE("rhsSubstraction"){
    FP<s64, SINGLE_PRECISION> fp = 11.536f;
	REQUIRE(Approx(11.536f - 2.585f).epsilon(ERROR_PRECISION) == (fp - 2.585));
}


TEST_CASE("lhsSubstraction"){
    FP<s64, SINGLE_PRECISION> fp = 11.536f;
	REQUIRE((2.5f - 11.536f) == (2.5f - fp));
}

/** Mixed fixed point multiplication different precision 
 * fo all fixed point operations with different precisions 
 * the result will take the largest representation between the 
 * two numbers
 * */ 

TEST_CASE("mixedFpMultiplication"){
    FP<s64, SINGLE_PRECISION> fp1 = 1.5f; 
    FP<s32, 16> fp2 = 2.5f;
	REQUIRE((fp1 * fp2) == (1.5f * 2.5f));
}


/** Constructor*/
TEST_CASE("constructor"){
    FP<s64, SINGLE_PRECISION> fp = 1.5f;
	REQUIRE(1.5f == fp);
}

TEST_CASE("setIntVal"){
    FP<s64, SINGLE_PRECISION> fp1;
    fp1.setIntValue(123456789);
	REQUIRE(fp1.getIntValue() == 123456789);
}

TEST_CASE("FloatValue"){
    FP<s64, SINGLE_PRECISION> fp;
    fp.setIntValue(123456789);
	REQUIRE( Approx(123456789/pow(2,SINGLE_PRECISION)).epsilon(ERROR_PRECISION) == fp);
}

TEST_CASE("createFromDifferentPrecision"){
    FP<s32, 16u> fp1(15.693f);
    FP<s64, SINGLE_PRECISION> fp2(fp1);
	REQUIRE(fp1 == fp2);
}

/** Create from int */
TEST_CASE("createFromint32"){
    s32 number = -141536;
    FP<s64, SINGLE_PRECISION> fp(number);
    REQUIRE(fp == number);
}
    
/** Create from unsigned int */
TEST_CASE("createFromUint32"){
    u32 number = 3975366;
    FP<s64, SINGLE_PRECISION> fp(number);
    REQUIRE(fp == number);
}

/** Create from long int */
TEST_CASE("createFromLongInt"){
    slng_t number = -396975366;
    FP<s64, SINGLE_PRECISION> fp(number);
    REQUIRE(fp == number);
}

/** Create from unsigned long int */
TEST_CASE("createFromULongInt"){
    ulng_t number = 353661259;
    FP<s64, SINGLE_PRECISION> fp(number);
    REQUIRE(fp == number);
}

/** Create from float */
TEST_CASE("createFromFloat"){
    float number = 3568.9634f;
    FP<s64, SINGLE_PRECISION> fp(number);
    REQUIRE(fp == number);
}

/** Create from double */
TEST_CASE("createFromDouble"){
    double number = 35684.9634;
    FP<s64, SINGLE_PRECISION> fp(number);
    REQUIRE(fp == number);
}

#if (defined(ARMA_USE_U64S64) || defined(ARMA_USE_U64S64_fixed_point_test))
/** Create from LL */
TEST_CASE("createFromLL"){
    s64 number = -35685554;
    FP<s64, SINGLE_PRECISION> fp(number);
    REQUIRE(fp == number);
}

/** Create from ULL */
TEST_CASE("createFromULL"){
    u64 number = 35685554;
    FP<s64, SINGLE_PRECISION> fp(number);
    REQUIRE(fp == number);
}
#endif

#if defined(ARMA_USE_U128S128)
/** Create from 128 bit integer */
TEST_CASE("createFrom128INT"){
    s128 number = -582745685554;
    FP<s128, SINGLE_PRECISION> fp(number);
    REQUIRE(fp == (double)number);
}

/** Create from U128 integer */
TEST_CASE("createFromU128INT"){
    u128 number = 582745685554;
    FP<u128, SINGLE_PRECISION> fp(number);
    REQUIRE(fp == (double)number);
}

#endif

/** Cast to other FP and other base type */
TEST_CASE("castToAnotherFixedPointType"){
    FP<s64, SINGLE_PRECISION> fp(168.36598f);
    FP<s32, 16u> fp_cast = static_cast<FP<s32, 16u> >(fp);
    REQUIRE(fp_cast == 168.36598f);
}

/** Assignment of FP different precision and different base type */
TEST_CASE("AssignementDifferentFixedPointPrecisions"){
    FP<s64, SINGLE_PRECISION> fp64_23(168.36598f);
    FP<s32, 16u> fp32_16 = static_cast<FP<s32, 16u> >(fp64_23);
    REQUIRE(fp64_23 == fp32_16);
}

/** Assignment of FP */
TEST_CASE("AssignementOfFixedPoint"){
    FP<s64, SINGLE_PRECISION> fp1(3568.136598f);
    FP<s64, SINGLE_PRECISION> fp2 = fp1;
    REQUIRE(fp1 == fp2);
}


/** Assignment of int32 */
TEST_CASE("AssignementOfInt32"){
    s32 number = -356987;
    FP<s64, SINGLE_PRECISION> fp = number;
    REQUIRE(fp == number);
}

/** Assignment of unsigned int32  */
TEST_CASE("AssignementOfUInt32"){
    u32 number = 35698736;
    FP<s64, SINGLE_PRECISION> fp = number;
    REQUIRE(fp == number);
}   

#if (defined(ARMA_USE_U64S64) || defined(ARMA_USE_U64S64_fixed_point_test))
/** Assignment of int64 */
TEST_CASE("AssignementOfInt64"){
    s64 number = -35698736;
    FP<s64, SINGLE_PRECISION> fp = number;
    REQUIRE(fp == number);
}   

/** Assignment of unsigned int64  */
TEST_CASE("AssignementOfUInt64"){
    u64 number = 356958736;
    FP<s64, SINGLE_PRECISION> fp = number;
    REQUIRE(fp == number);
}   

#endif

#if defined(ARMA_USE_U128S128)
 /** Assignment of int128 */
TEST_CASE("AssignementOfInt128"){
    s128 number = -5899535698736;
    FP<s128, SINGLE_PRECISION> fp = number;
    REQUIRE(fp == (double)number);
}   

/** Assignment of unsigned int128  */
TEST_CASE("AssignementOfUInt128"){
    u128 number = 34523668945698736;
    FP<s128, SINGLE_PRECISION> fp = number;
    REQUIRE(fp == (double)number); // cast to double because error: ambiguous overload for ‘operator<<’ catch.hpp:1549:49:
}   
#endif

/** Assignment of float */
TEST_CASE("AssignementOfFloat"){
    float number = 3652.98545f;
    FP<s64, SINGLE_PRECISION> fp = number;
    REQUIRE(fp == number);
}

/** Assignment of double */
TEST_CASE("AssignementOfDouble"){
    const double number = 365289.98545;
    FP<s64, SINGLE_PRECISION> fp = number;
    REQUIRE(fp == number);
}

/** Addition assignment */
TEST_CASE("AdditionAssignment"){
    FP<s64, SINGLE_PRECISION> fp1 = 35.335f;
    FP<s64, SINGLE_PRECISION> fp2 = 115.189f;
    fp1 += fp2;
    REQUIRE(fp1 == (35.335f + 115.189f));
}
 
/** Addition assignment with other types Example : float number*/
TEST_CASE("AdditionAssignemenWithOtherTypes"){
    const float number = 365289.985450f;
    FP<s64, SINGLE_PRECISION> fp = 35.335;
    fp += number;
    REQUIRE(fp == (35.335 + number));
}

/** Minus operator*/
TEST_CASE("MinusOperator"){
    FP<s64, SINGLE_PRECISION> fp = 115;
    REQUIRE(-fp == -115);
}

/** Subtraction assignment */
TEST_CASE("SubstractionAssignment"){
    FP<s64, SINGLE_PRECISION> fp1 = 35.335f;
    FP<s64, SINGLE_PRECISION> fp2 = 115.189f;
    fp1 -= fp2;
    REQUIRE(fp1 == (35.335f - 115.189f));
}
    
/** Subtraction */ 
TEST_CASE("Substraction"){
    FP<s64, SINGLE_PRECISION> fp1 = 10.0f;
    FP<s64, SINGLE_PRECISION> fp2 = 2.5f;
	REQUIRE((fp1 - fp2) == (10.0f - 2.5f));
}

/** Subtraction assignment with other types Example : float number*/
TEST_CASE("SubstractionAssignemenWithOtherTypes"){
    const float number = 389.985f;
    FP<s64, SINGLE_PRECISION> fp = 35.335;
    fp -= number;
    REQUIRE(Approx(35.335 - number) == fp);
}

/** Subtraction with other types Example : double*/
TEST_CASE("SubstractionWithOtherTypes"){
#if defined(ARMA_USE_U128S128)
    const double number = 3289.8500488489653247;
    FP<s128, DOUBLE_PRECISION> fp = 35.335;
    FP<s128, DOUBLE_PRECISION> res;
    res = fp - number;
    REQUIRE(Approx(35.335 - number).epsilon(ERROR_DPRECISION) == res);
#endif
}

/** Multiplication assignment */
TEST_CASE("MultiplicationAssignment"){
    FP<s64, SINGLE_PRECISION> fp1 = 35.335f;
    FP<s64, SINGLE_PRECISION> fp2 = 115.189f;
    fp1 *= fp2;
    REQUIRE(Approx(35.335f * 115.189f) == fp1);
}

/** Multiplication */
TEST_CASE("MultiplicationFixedWithFixed"){
    FP<s64, SINGLE_PRECISION> fp1(11.53f);
    FP<s64, SINGLE_PRECISION> fp2(2.51f);
	REQUIRE( Approx(11.53f * 2.51f) == (fp1 * fp2));
}   

/** Multiplication with other precision types*/
TEST_CASE("MultiplicationWithOtherPrecisionTypes"){
	FP<s64, SINGLE_PRECISION> fp64_23(168.368048f);
    FP<s32, 16u> fp32_16 (15.036);
  	FP<s64, SINGLE_PRECISION> res = fp64_23 * static_cast<FP<s64, SINGLE_PRECISION> >(fp32_16);
    REQUIRE(Approx(168.368048f * 15.036) == res);
}   

/** Multiplication with float */
TEST_CASE("MultiplicationWithFloat"){
    const float number = 1236.458f;
    FP<s64, SINGLE_PRECISION> fp(2.51f);
    REQUIRE(Approx(2.51f * number) == (fp * number));
}   

/** Multiplication with double */
TEST_CASE("MultiplicationWithDouble"){
#if defined(ARMA_USE_U128S128) 
    const double number = 1236.458589561235698;
    FP<s128, DOUBLE_PRECISION> fp(2.51f);
    REQUIRE((double)(fp * number) == Approx(2.51f * number));
#endif    
}   

/** Multiplication with int32 */
TEST_CASE("MultiplicationWithInt32"){
    const s32 number = -1236;
    FP<s64, SINGLE_PRECISION> fp(2.51f);
    REQUIRE((float)(fp * number) == Approx(2.51f * number));
}   

/** Multiplication with uint32 */
TEST_CASE("MultiplicationWithUint32"){
    const u32 number = 1236;
    FP<s64, SINGLE_PRECISION> fp(2.51f);
    REQUIRE((float)(fp * number) == Approx(2.51f * number));
}


#if (defined(ARMA_USE_U64S64) || defined(ARMA_USE_U64S64_fixed_point_test))
/** Multiplication with int64 */
TEST_CASE("MultiplicationWithInt64"){
    const s64 number = -1236;
    FP<s64, SINGLE_PRECISION> fp(2.51f);
    REQUIRE((float)(fp * number) == Approx(2.51f * number));
}   

/** Multiplication with uint64 */
TEST_CASE("MultiplicationWithUint64"){
    const u64 number = 1236;
    FP<s64, SINGLE_PRECISION> fp(2.51f);
	REQUIRE((float)(fp * number) == Approx(2.51f * number));
}
#endif

#if defined(ARMA_USE_U128S128)
/** Multiplication with int128 */
TEST_CASE("MultiplicationWithInt128"){
    const s128 number = -1236;
    FP<s64, SINGLE_PRECISION> fp(2.51f);
	REQUIRE((double)(fp * number) == Approx(2.51f * number));
}   

/** Multiplication with uint128 */
TEST_CASE("MultiplicationWithUint128"){
    const u128 number = 1236;
    FP<s64, SINGLE_PRECISION> fp(2.51f);
	REQUIRE((double)(fp * number) == Approx(2.51f * number));
}
#endif


/** Lhs multiplication by other types Example : float */
TEST_CASE("LhsMultiplication"){
    const float number = 3.6902f;
    FP<s64, SINGLE_PRECISION> fp = 200.3156f;
	REQUIRE((float)(number * fp) == Approx(number * 200.3156f));
}

/** Rhs multiplication by other types Example : float */
TEST_CASE("RhsMultiplication"){
    const float number = 3.6902f;
    FP<s64, SINGLE_PRECISION> fp = 200.3156f;
	REQUIRE((float)(fp * number) == Approx(200.3156f * number));
}

/** Division assignment */
TEST_CASE("DivisionAssignment"){
    const float number = 1.365f;
    FP<s64, SINGLE_PRECISION> fp = 189.523658f;
    fp /= number;
	REQUIRE(Approx(189.523658f / 1.365f) == fp);
}

/** Normal division */
TEST_CASE("NormalDivision"){
    FP<s64, SINGLE_PRECISION> fp1 = 1.51f;
    FP<s64, SINGLE_PRECISION> fp2 = 2.505f;
	REQUIRE((float)(fp1/fp2) == Approx(1.51f / 2.505f));
}

/** Division by other types Example : Double */
TEST_CASE("DivisionByOtherTypes"){
#if defined(ARMA_USE_U128S128)
    const double number = 1.365f;
    FP<s128, DOUBLE_PRECISION> fp = 17489.5236581236589745;
    fp = fp / number;
	REQUIRE(Approx(17489.5236581236589745 / 1.365f) == fp);
#endif    
}

/** Lhs division by other types Example : float */
TEST_CASE("LhsDivision"){
    const float number = 20.563f;
    FP<s64, SINGLE_PRECISION> fp = 2.5f;
	REQUIRE(Approx(number / 2.5f) == (number / fp));
}

/** Rhs division by other types Example : float */
TEST_CASE("RhsDivision"){
    const float number = 20.563f;
    FP<s64, SINGLE_PRECISION> fp = 11.155f;
	REQUIRE(Approx(11.155f / number) == (fp / number));
}

/** Equality operator */
TEST_CASE("EqualityWithOtherType"){
    const float number = 20.563f;
    FP<s64, SINGLE_PRECISION> fp(number);
	REQUIRE(fp == number);
}

/** Different operator*/
TEST_CASE("InequalityOperator"){
    const float number = 20.563f;
    FP<s64, SINGLE_PRECISION> fp(12.3);
	REQUIRE(fp != number);
}

/** Smaller than */
TEST_CASE("SmallerThanOperator"){
    FP<s64, SINGLE_PRECISION> fp1(20.563f);
    FP<s64, SINGLE_PRECISION> fp2(12.3);
	REQUIRE(fp2 < fp1);
}

/** Bitwise shift up assignment */
TEST_CASE("BitwiseShiftUpAssignment"){
    FP<s64, SINGLE_PRECISION> fp(256);
    fp <<= 3;
	REQUIRE((256 * pow(2, 3)) == fp);
}

/** Bitwise shift up */
TEST_CASE("BitwiseShiftUp"){
    FP<s64, SINGLE_PRECISION> fp(256);
    FP<u64, SINGLE_PRECISION> res;
    res = fp << 3;
	REQUIRE((256 * pow(2, 3)) == res);
}

//******************************************************************************//
//*     This part concerns unit tests for changes added to armadillo           *//
//*****************************************************************************//
fmat float_matrix_a, float_matrix_b;
cx_fmat cx_float_matrix_a, cx_float_matrix_b;
    
fixed_fmat fxMatA = randu<fixed_fmat>(NBROWS, NBCOLS);
fixed_fmat fxMatB = randu<fixed_fmat>(NBROWS, NBCOLS);

cx_fixed_fmat cx_fxMatA = randu<cx_fixed_fmat>(NBROWS, NBCOLS);
cx_fixed_fmat cx_fxMatB = randu<cx_fixed_fmat>(NBROWS, NBCOLS);
   
   
/**  Copy from fixed point matrix to float/double matrix */
TEST_CASE("CopyToRealMat"){
	float_matrix_a = fxMatA;
	cx_float_matrix_a = cx_fxMatA;
    fmat res = float_matrix_a - static_cast<fmat>(fxMatA);
    REQUIRE(0.0f == abs(res).max());
    
    cx_fmat cx_res = cx_float_matrix_a - static_cast<cx_fmat>(cx_fxMatA);
    REQUIRE(0.0f == abs(cx_res).max());

}

/** copy from float to fixed*/
TEST_CASE("copyFromFloatToFixed"){
	fmat matrix = randu<fmat>(NBROWS, NBCOLS);
	cx_fmat cx_matrix = randu<cx_fmat>(NBROWS, NBCOLS);
	
	fixed_fmat fixed_matrix = matrix;
	cx_fixed_fmat cx_fixed_matrix = cx_matrix;
	
    fmat res = abs(matrix) - abs(static_cast<fmat>(fixed_matrix));
    REQUIRE(Approx(0.0f).epsilon(ERROR_PRECISION) ==  res.max());
    
    fmat diff_abs_cx_res = abs(cx_matrix) - abs(static_cast<cx_fmat>(cx_fixed_matrix));
    REQUIRE(Approx(0.0f).epsilon(ERROR_PRECISION) ==  diff_abs_cx_res.max());
}

/** copy from double to fixed*/
TEST_CASE("CopyFromDoubleToFixed"){
	mat       matrix = randu<mat>(NBROWS, NBCOLS);
	cx_mat cx_matrix = randu<cx_mat>(NBROWS, NBCOLS);
	
	fixed_mat       fixed_matrix = matrix;
	cx_fixed_mat cx_fixed_matrix = cx_matrix;
	
    mat res = matrix - static_cast<mat>(fixed_matrix);
    REQUIRE(Approx(0.0).epsilon(ERROR_DPRECISION) == abs(res).max());
    
    cx_mat cx_res = cx_matrix - static_cast<cx_mat>(cx_fixed_matrix);
    REQUIRE(Approx(0.0).epsilon(ERROR_DPRECISION) == abs(cx_res).max());

}

/** Matrix multiplication */
TEST_CASE("Matrix_multiplication"){
    fixed_fmat fixed_multiplication_matrix = fxMatA * fxMatB;
    float_matrix_a = fxMatA; float_matrix_b = fxMatB;
    fmat multiplication_matrix = float_matrix_a * float_matrix_b;
    
    fmat res = abs(multiplication_matrix) - abs(static_cast<fmat>(fixed_multiplication_matrix));
    REQUIRE(Approx(0.0f).epsilon(ERROR_PRECISION) ==  res.max());
    
    //Test for complex
    cx_fixed_fmat cx_fixed_multiplication_matrix = cx_fxMatA * cx_fxMatB;
    cx_float_matrix_a = cx_fxMatA; cx_float_matrix_b = cx_fxMatB;
    cx_fmat cx_multiplication_matrix = cx_float_matrix_a * cx_float_matrix_b;
    
    fmat diff_cx_res = abs(cx_multiplication_matrix) - abs(static_cast<cx_fmat>(cx_fixed_multiplication_matrix));
    REQUIRE(Approx(0.0f).epsilon(ERROR_PRECISION) ==  diff_cx_res.max());
    
}

/** Matrix element-wise multiplication */
TEST_CASE("Element_wise_matrix_multiplication"){

    fixed_fmat fixed_multiplication_matrix = fxMatA % fxMatB;
    float_matrix_a = fxMatA; float_matrix_b = fxMatB;
    fmat multiplication_matrix = float_matrix_a % float_matrix_b;
    
    fmat res = abs(multiplication_matrix) - abs(static_cast<fmat>(fixed_multiplication_matrix));
    REQUIRE(Approx(0.0f).epsilon(ERROR_PRECISION) ==  res.max());
    
    //Test for complex
    cx_fixed_fmat cx_fixed_multiplication_matrix = cx_fxMatA % cx_fxMatB;
    cx_float_matrix_a = cx_fxMatA; cx_float_matrix_b = cx_fxMatB;
    cx_fmat cx_multiplication_matrix = cx_float_matrix_a % cx_float_matrix_b;
    
    fmat diff_cx_res = abs(cx_multiplication_matrix) - abs(static_cast<cx_fmat>(cx_fixed_multiplication_matrix));
    REQUIRE(Approx(0.0f).epsilon(ERROR_PRECISION) ==  diff_cx_res.max());
}

/** Matrix element-wise division */
TEST_CASE("Element_wise_matrix_division"){
    fixed_fmat fixed_division_matrix = fxMatA / fxMatB;
    float_matrix_a = fxMatA; float_matrix_b = fxMatB;
    fmat division_matrix = float_matrix_a / float_matrix_b;
    
    fmat res = abs(division_matrix) - abs(static_cast<fmat>(fixed_division_matrix));
    //res.print();
    REQUIRE(Approx(0.0f).epsilon(ERROR_PRECISION) ==  res.max());
    
      
    //Test for complex
    cx_fixed_fmat cx_fixed_division_matrix = cx_fxMatA / cx_fxMatB;
    cx_float_matrix_a = cx_fxMatA; cx_float_matrix_b = cx_fxMatB;
    cx_fmat cx_division_matrix = cx_float_matrix_a / cx_float_matrix_b;
    
    mat diff_cx_res = abs(cx_division_matrix) - abs(static_cast<cx_mat>(cx_fixed_division_matrix));
    REQUIRE(Approx(0.0f).epsilon(ERROR_PRECISION) ==  diff_cx_res.max());
    
}

/** Matrix addition */
TEST_CASE("Matrix_addition"){
    fixed_fmat fixed_addition_matrix = fxMatA + fxMatB;
    float_matrix_a = fxMatA; float_matrix_b = fxMatB;
    fmat addition_matrix = float_matrix_a + float_matrix_b;
    
    fmat res = abs(addition_matrix) - abs(static_cast<fmat>(fixed_addition_matrix));
    //res.print();
    REQUIRE(Approx(0.0f).epsilon(ERROR_PRECISION) ==  res.max());
    
     //Test for complex
    cx_fixed_fmat cx_fixed_addition_matrix = cx_fxMatA + cx_fxMatB;
    cx_float_matrix_a = cx_fxMatA; cx_float_matrix_b = cx_fxMatB;
    cx_fmat cx_addition_matrix = cx_float_matrix_a + cx_float_matrix_b;
    
    fmat diff_cx_res = abs(cx_addition_matrix) - abs(static_cast<cx_fmat>(cx_fixed_addition_matrix));
    REQUIRE(Approx(0.0f).epsilon(ERROR_PRECISION) ==  diff_cx_res.max());
}

/** Matrix Subtraction  */
TEST_CASE("Matrix_subtraction"){
    fixed_fmat fixed_subtraction_matrix = fxMatA - fxMatB;
    float_matrix_a = fxMatA; float_matrix_b = fxMatB;
    fmat subtraction_matrix = float_matrix_a - float_matrix_b;
    
    fmat res = abs(subtraction_matrix) - abs(static_cast<fmat>(fixed_subtraction_matrix));
    //res.print();
    REQUIRE(Approx(0.0f).epsilon(ERROR_PRECISION) ==  res.max());
    
     //Test for complex
    cx_fixed_fmat cx_fixed_subtraction_matrix = cx_fxMatA - cx_fxMatB;
    cx_float_matrix_a = cx_fxMatA; cx_float_matrix_b = cx_fxMatB;
    cx_fmat cx_subtraction_matrix = cx_float_matrix_a - cx_float_matrix_b;
    
    fmat diff_cx_res = abs(cx_subtraction_matrix) - abs(static_cast<cx_fmat>(cx_fixed_subtraction_matrix));
    REQUIRE(Approx(0.0f).epsilon(ERROR_PRECISION) ==  diff_cx_res.max());
}

/** Matrix inversion  */
TEST_CASE("Matrix_inversion"){
    arma_rng::set_seed_random();
    //Test for non complex
    fixed_fmat       _fixed_mat = randu<fixed_fmat>(NBROWS, NBCOLS);
    cx_fixed_fmat _cx_fixed_mat = randu<cx_fixed_fmat>(NBROWS, NBCOLS);


    fixed_fmat identity_fixed_fmat = inv(_fixed_mat) * _fixed_mat;
    fixed_fmat identity = eye<fixed_fmat>(NBROWS, NBCOLS);
    
    fixed_fmat res = identity - identity_fixed_fmat;    
    REQUIRE(Approx(0.0f).epsilon(0.001) == abs(res).max());
    
    //Test for complex
    cx_fixed_fmat cx_identity_fixed_fmat = inv(_cx_fixed_mat) * _cx_fixed_mat;
    cx_fixed_fmat cx_identity = eye<cx_fixed_fmat>(NBROWS, NBCOLS);

    cx_fixed_fmat cx_res = cx_identity - cx_identity_fixed_fmat;
    REQUIRE(Approx(0.0f).epsilon(0.001) ==  abs(cx_res).max());
}

/** Construct matrix from submatrix different type */
TEST_CASE("submatrix_from_matrix_different_type"){
    
    uvec rows; rows << 0 << 1 << 2 << 3 << 4;
    uvec cols; cols << 0 << 1 << 2 << 3 <<4;
    
    mat b = fxMatA.submat(rows,cols);
    
    for(unsigned int i=0; i < rows.size(); i++)
    for(unsigned int j=0; j < cols.size(); j++)
        REQUIRE(0 == b(i,j)-fxMatA(i,j));
    
}

/** Construct matrix from matrix different type */
TEST_CASE("matrix_from_matrix_different_type"){
    
    // fixed to float or double (real & complex)
    mat  a = fxMatA;
    fmat b = fxMatB;
    
    cx_mat cx_a = cx_fxMatA;
    cx_fmat cx_b = cx_fxMatB;
    
    REQUIRE(0.0 == abs(a -fxMatA).max());
    REQUIRE(0.0f == abs(b - fxMatB).max());
    
    REQUIRE(0.0 == abs(cx_a - static_cast<cx_mat>(cx_fxMatA)).max());
    REQUIRE(0.0f == abs(cx_b - static_cast<cx_fmat>(cx_fxMatB)).max());
    
    // float or double to fixed (real & complex)
    fixed_fmat fixeda = a;
    fixed_fmat fixedb = b;
    
    cx_fixed_fmat cx_fxa = cx_a;
    cx_fixed_fmat cx_fxb = cx_b;
    
    REQUIRE(0.0 == abs(fixeda - a).max());
    REQUIRE(0.0f == abs(fixedb - b).max());
    
    REQUIRE(0.0 == abs(cx_a - static_cast<cx_mat>(cx_fxa)).max());
    REQUIRE(0.0f == abs(cx_b - static_cast<cx_fmat>(cx_fxb)).max());
    
}

/** Construct a matrix from a given auxiliary array different type */
TEST_CASE("matrix_pointer_of_data_different_type"){
    unsigned int n_rows = 4;
    unsigned int n_cols = 4;
    
    //Fixed matrix from array of reals
    float * ptr_aux_mem_float = new float[n_rows * n_cols];
    
    for(unsigned int i=0; i < n_rows * n_cols; i++)
        ptr_aux_mem_float[i] = (rand() % 100)+1;
        
    fixed_fmat fx_matrix = fixed_fmat(ptr_aux_mem_float, n_rows, n_cols);
    
    for(unsigned int i=0; i < n_rows * n_cols; i++)
        REQUIRE(ptr_aux_mem_float[i] == fx_matrix.mem[i]);
    
    delete[] ptr_aux_mem_float;
    
    // Real matrix from array of fixed
    fixed64_23 * ptr_aux_mem = new fixed64_23[n_rows * n_cols];
    
    for(unsigned int i=0; i < n_rows * n_cols; i++)
        ptr_aux_mem[i] = (rand() % 100)+1;
    mat matrix = mat(ptr_aux_mem, n_rows, n_cols);
    
    for(unsigned int i=0; i < n_rows * n_cols; i++)
        REQUIRE(ptr_aux_mem[i] == matrix.mem[i]);
    
    delete[] ptr_aux_mem;    
}

/** Construct a row & col vector from a given auxiliary array different type */
TEST_CASE("row_col_from_pointer_of_data_different_type"){
    unsigned int n_elem = 10;
        
    fixed64_23 * ptr_aux_mem_fixed = new fixed64_23[n_elem];
    float* ptr_aux_mem_float = new float [n_elem];
    
    for(unsigned int i = 0; i < n_elem; i++){
        ptr_aux_mem_fixed[i] = (rand() % 100)+1;
        ptr_aux_mem_float[i] = (rand() % 100)+1;
    }
    
    // Create fixed Row & Col from float auxiliary array
    Row<fixed64_23> row_fixed = Row<fixed64_23>(ptr_aux_mem_float, n_elem);    
    Col<fixed64_23> col_fixed = Col<fixed64_23>(ptr_aux_mem_float, n_elem);
    
    
    // Create float, double Row & Col from fixed auxiliary array
    Col<float>  col_float  = Col<float>(ptr_aux_mem_fixed, n_elem);
    Col<double> col_double = Col<double>(ptr_aux_mem_fixed, n_elem);
    Row<float>  row_float  = Row<float>(ptr_aux_mem_fixed, n_elem);
    Row<double> row_double = Row<double>(ptr_aux_mem_fixed, n_elem);
    
    for(unsigned int i = 0; i < n_elem; i++){
        REQUIRE(row_fixed[i]  == ptr_aux_mem_float[i]);
        REQUIRE(col_fixed[i]  == ptr_aux_mem_float[i]);
        REQUIRE(col_float[i]  == ptr_aux_mem_fixed[i]);
        REQUIRE(col_double[i] == ptr_aux_mem_fixed[i]);
        REQUIRE(row_float[i]  == ptr_aux_mem_fixed[i]);
        REQUIRE(row_double[i] == ptr_aux_mem_fixed[i]);
    }
    delete[] ptr_aux_mem_fixed;
    delete[] ptr_aux_mem_float;
}

/** Construct matrix from string */
TEST_CASE("construct_fixed_point_matrix_from_string"){
    mat A = 
    "\
     0.061198   0.201990   0.019678  -0.493936  -0.126745   0.051408;\
     0.437242   0.058956  -0.149362  -0.045465   0.296153   0.035437;\
    -0.492474  -0.031309   0.314156   0.419733   0.068317  -0.454499;\
     0.336352   0.411541   0.458476  -0.393139  -0.135040   0.373833;\
     0.239585  -0.428913  -0.406953  -0.291020  -0.353768   0.258704;\
    ";
    
    fixed_fmat fA = 
    "\
     0.061198   0.201990   0.019678  -0.493936  -0.126745   0.051408;\
     0.437242   0.058956  -0.149362  -0.045465   0.296153   0.035437;\
    -0.492474  -0.031309   0.314156   0.419733   0.068317  -0.454499;\
     0.336352   0.411541   0.458476  -0.393139  -0.135040   0.373833;\
     0.239585  -0.428913  -0.406953  -0.291020  -0.353768   0.258704;\
    ";
    
    for(unsigned int i = 0; i< A.n_rows; i++)
    for(unsigned int j = 0; j< A.n_cols; j++)
        REQUIRE(Approx(A(i, j)) == fA(i, j));
}
