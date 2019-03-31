/// <summary>
/// Contains definition for the class axis::foundation::metaprogramming::Factorial.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once

namespace axis
{
	namespace foundation
	{
		namespace metaprogramming
		{

			/// <summary>
			/// Calculates at compile-time the factorial of an positive integer number.
			/// </summary>
			template<int number>
			struct Factorial
			{
				enum { 
					/// <summary>
					/// Contains the power result.
					/// </summary>
					value = number * Factorial<number - 1> 
				};
			};

			/// <summary>
			/// Calculates at compile-time the power of an integer number.
			/// </summary>
			template<int number>
			struct Factorial<0>
			{
				enum { 
					/// <summary>
					/// Contains the power result.
					/// </summary>
					value = 1
				}
			};

		}
	}
}
