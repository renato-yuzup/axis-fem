#include "matrix_operations.hpp"
#include <math.h>
#include "linear_algebra.hpp"

#define test_func(xi,i1,i3) (2*(xi)*(xi)*(xi) - (i1)*(xi)*(xi) + (i3))
#define PI 3.1415926535897932384626433832795
#define TOLERANCE 1.0e-6

namespace ayfb = axis::yuzu::foundation::blas;

namespace {

  GPU_ONLY inline bool is_equalf(real x1, real x2)
  {
    real diff = abs(x1*x1 - x2*x2);
    diff *= diff;
    return sqrt(diff) <= TOLERANCE;
  }

  GPU_ONLY inline int count_equal(const real x[], int valueCount)
  {
    int n = 0;
    for (int i = 0; i < valueCount; i++)
    {
      for (int j = i + 1; j < valueCount; j++)
      {
        if (is_equalf(x[i], x[j])) n++;
      }
    }
    return n;
  }

} // namespace

GPU_ONLY void ayfb::SymmetricEigen( real *eigenValues,  
  DenseMatrix *eigenProjections[], int& eigenCount, const SymmetricMatrix& m )
{
  /*
   * The algorithm below is based on Box A.5 of Souza Netto et al (2008),
   * Computational Methods for Plasticity: Theory and Applications. Wiley.
  */
  // Compute matrix invariants
  real i1 = m.Trace();
  real i2 = m(0,0)*m(1,1) + m(1,1)*m(2,2) + m(0,0)*m(2,2) 
    - m(0,1)*m(0,1) - m(0,2)*m(0,2) - m(1,2)*m(1,2);
  real i3 = Determinant3D(m);

  // Compute coefficients
  real x[3]; // eigenvalues
  real R = (-2*i1*i1*i1 + 9*i1*i2-27*i3) / 54.0;
  real Q = abs((i1*i1 - 3*i2) / 9.0);
  real Q3squared = sqrt(Q*Q*Q);
  if (Q3squared == 0) // all eigenvalues are equal
  {
    x[0] = x[1] = x[2] = i1 / 3.0;
  }
  else
  {
    // we have at least two different eigenvalues
    real aux = R / Q3squared;
    real phi = 0;
    if (aux >= 1.0) // branch to avoid rounding error
    {
      phi = 1 - 9.9999995e-1;
    }
    else if (aux <= -1.0)
    {
      phi = PI * 9.9999995e-1;
    }
    else
    {
      phi = acos(aux);
    }

    // Calculate eigenvalues
    real p1 = -2*sqrt(Q);
    real p2 = i1 / 3.0;
    x[0] = p1*cos(phi/3.0) + p2;
    x[1] = p1*cos((phi + 2*PI) / 3.0) + p2;
    x[2] = p1*cos((phi - 2*PI) / 3.0) + p2;
  }

  // Compute eigenprojections
  eigenCount = 3 - count_equal(x, 3);
  if (eigenCount == 1)
  {
    eigenValues[0] = x[0];
    Identity3D(*eigenProjections[0]);
  }
  else
  {
    const real xi = x[0];
    eigenValues[0] = xi;
    DenseMatrix& Ei = *eigenProjections[0];
    real denom = (2*xi*xi*xi - i1*xi*xi + i3);
      real c1 = xi / denom;
      real c2 = i3 / xi;
      Product(Ei, 1.0, m, m);
      Sum(Ei, 1.0, Ei, -(i1 - xi), m);
      Ei(0,0) += c2; Ei(1,1) += c2; Ei(2,2) += c2;
      Ei.Scale(c1);

    if (eigenCount == 2)
    {
      eigenValues[1] = is_equalf(x[0], x[1])? x[2] : x[1];
      DenseMatrix& Ej = *eigenProjections[1];
      Ej.CopyFrom(Ei); Ej *= -1.0;
      Ej(0,0) += 1.0; Ej(1,1) += 1.0; Ej(2,2) += 1.0; 
    }
    else
    {
      for (int j = 1; j < 3; j++)
      {
        const real xj = x[j];
        eigenValues[j] = xj;
        DenseMatrix& Ej = *eigenProjections[j];
        real c1 = xj / (2*xj*xj*xj - i1*xj*xj + i3);
        real c2 = i3 / xj;
        Product(Ej, 1.0, m, m);
        Sum(Ej, 1.0, Ej, -(i1 - xj), m);
        Ej(0,0) += c2; Ej(1,1) += c2; Ej(2,2) += c2;
        Ej.Scale(c1);
      }
    }
  }
}

GPU_ONLY void ayfb::SymmetricLogarithm( SymmetricMatrix& lhr, 
  const SymmetricMatrix& m )
{
  real eigs[3];
  DenseMatrix e0(3,3), e1(3,3), e2(3,3);
  DenseMatrix *e[] = {&e0, &e1, &e2};
  int eigsCount = 0;
  SymmetricEigen(eigs, e, eigsCount, m);

  lhr.ClearAll();
  for (int i = 0; i < eigsCount; i++)
  {
    DenseMatrix& Ei = *e[i];
    const real xi = eigs[i];
    const real yi = log(xi);
    Sum(lhr, 1.0, lhr, yi, Ei);
  }
}

GPU_ONLY void ayfb::SymmetricSquareRoot( SymmetricMatrix& lhr, 
  const SymmetricMatrix& m )
{
  real eigs[3];
  DenseMatrix e0(3,3), e1(3,3), e2(3,3);
  DenseMatrix *e[] = {&e0, &e1, &e2};
  int eigsCount = 0;
  SymmetricEigen(eigs, e, eigsCount, m);

  lhr.ClearAll();
  for (int i = 0; i < eigsCount; i++)
  {
    DenseMatrix& Ei = *e[i];
    const real xi = eigs[i];
    const real yi = sqrt(xi);
    Sum(lhr, 1.0, lhr, yi, Ei);
  }
}

GPU_ONLY void ayfb::SymmetricExponential( SymmetricMatrix& lhr, 
  const SymmetricMatrix& m )
{
  real eigs[3];
  DenseMatrix e0(3,3), e1(3,3), e2(3,3);
  DenseMatrix *e[] = {&e0, &e1, &e2};
  int eigsCount = 0;
  SymmetricEigen(eigs, e, eigsCount, m);

  lhr.ClearAll();
  for (int i = 0; i < eigsCount; i++)
  {
    DenseMatrix& Ei = *e[i];
    const real xi = eigs[i];
    const real yi = exp(xi);
    Sum(lhr, 1.0, lhr, yi, Ei);
  }
}
