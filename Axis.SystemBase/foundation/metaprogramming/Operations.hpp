/// <summary>
/// Contains definition for the class axis::foundation::metaprogramming::Operation and its subclasses.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "Power.hpp"

namespace axis
{
	namespace foundation
	{
		namespace metaprogramming
		{

			/// <summary>
			/// Describes the basic operation where no operation is executed.
			/// </summary>
			struct NoOp
			{
				enum { 
					/// <summary>
					/// Contains the numerical identifier of the basic operation.
					/// </summary>
					value = 1
				};
			};

			/// <summary>
			/// Describes a generic arithmetic operation.
			/// </summary>
			/// <typeparam name="id">The numerical identifier of this operation.</typeparam>
			/// <typeparam name="InnerOp">The operation which will run on the operands before this operation.</typeparam>
			template<int id, class InnerOp = NoOp>
			struct Operation
			{
				enum { 
					/// <summary>
					/// Contains the numerical identifier of the composite operation.
					/// </summary>
					value = Power<id, InnerOp::value>::value
				};
			};

			/// <summary>
			/// Specifies a sum of two operands.
			/// </summary>
			/// <typeparam name="InnerOp">The operation which will run on the operands before this operation.</typeparam>
			template<class InnerOp = NoOp>
			struct Sum : Operation<2, InnerOp>
			{
			};

			/// <summary>
			/// Specifies a subtraction of two operands.
			/// </summary>
			/// <typeparam name="InnerOp">The operation which will run on the operands before this operation.</typeparam>
			template<class InnerOp = NoOp>
			struct Subtraction : Operation<3, InnerOp>
			{
			};


			/// <summary>
			/// Specifies a multiplication of two operands.
			/// </summary>
			/// <typeparam name="InnerOp">The operation which will run on the operands before this operation.</typeparam>
			template<class InnerOp = NoOp>
			struct Product : Operation<5, InnerOp>
			{
			};


			/// <summary>
			/// Specifies a division of two operands.
			/// </summary>
			/// <typeparam name="InnerOp">The operation which will run on the operands before this operation.</typeparam>
			template<class InnerOp = NoOp>
			struct Division : Operation<7, InnerOp>
			{
			};
		}
	}
}
