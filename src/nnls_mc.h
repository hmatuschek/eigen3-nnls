#ifndef __EIGEN_NNLS_MC_H__
#define __EIGEN_NNLS_MC_H__

#include "nnls.h"
namespace Eigen {

	/**
	* \brief A multi-column wrap function of a NNLS solver.
	* \param mna the m by n matrix A.
	* \param mpb the m by p matrix b.
	* \param npx the n by p matrix x, which need not be initialized.
	* \param m the rows of A.
	* \param n the columns of A and the rows of x.
	* \param p the columns of b and the columns of x.
	* \param eps Specifies the precision of the optimum..	
	* \return true on success and false otherwise.
	*/
	template <typename MatrixType>
	bool nnls_mc(typename MatrixType::Scalar *mna, typename MatrixType::Scalar* mpb, typename MatrixType::Scalar* npx, int m, int n, int p, typename MatrixType::Scalar eps = 1e-10)
	{
		MatrixType A(m,n);
		MatrixType b(m,1);

		for(int i=0; i<m; i++)
			for(int j=0; j<n; j++)
				A(i,j) = mna[i*n+j];

		for(int j=0; j<p; j++)
		{
			for(int i=0; i<m; i++)
				b(i) = mpb[i*p+j];

			NNLS<MatrixType> nnls(A, 30, eps);

			if (! nnls.solve(b))
				return false;

			for(int i=0; i<n; i++)
				npx[i*p+j] = nnls.x()(i);
		}

		return true;
	}

	/**
	* \brief A multi-column wrap function of a Least Square solver.
	* This function is just for result comparison.
	* \param mna the m by n matrix A.
	* \param mpb the m by p matrix b.
	* \param npx the n by p matrix x, which need not be initialized.
	* \param m the rows of A.
	* \param n the columns of A and the rows of x.
	* \param p the columns of b and the columns of x.
	* \param eps Specifies the precision of the optimum..	
	* \return true on success and false otherwise.
	*/
	template <typename MatrixType>
	bool ls_mc(typename MatrixType::Scalar *mna, typename MatrixType::Scalar* mpb, typename MatrixType::Scalar* npx, int m, int n, int p, typename MatrixType::Scalar eps = 1e-10)
	{
		MatrixType A(m,n);
		MatrixType b(m,p);

		for(int i=0; i<m; i++)
		{
			for(int j=0; j<n; j++)
				A(i,j) = mna[i*n+j];

			for(int j=0; j<p; j++)
				b(i,j) = mpb[i*p+j];
		}

		MatrixType res = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b) ;

		for(int i=0; i<n; i++)
			for(int j=0; j<p; j++)
				npx[i*p+j] = res(i,j);

		return true;
	}
}
#endif // __EIGEN_NNLS_MC_H__
