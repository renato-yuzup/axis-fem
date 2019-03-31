/// <summary>
/// Contains definition for the class axis::foundation::metaprogramming::Power.
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
			/// Calculates at compile-time the power of an integer number.
			/// </summary>
			template<int base, int exponent>
			struct Power 
			{
				enum { 
					/// <summary>
					/// Contains the power result.
					/// </summary>
					value = base * Power<base, exponent - 1>::value
				};
			};

			/// <summary>
			/// Calculates at compile-time the power of an integer number.
			/// </summary>
			template<int base>
			struct Power<base, 1>
			{
				enum { 
					/// <summary>
					/// Contains the power result.
					/// </summary>
					value = base
				};
			};

			/// <summary>
			/// Calculates at compile-time the power of an integer number.
			/// </summary>
			template<int base>
			struct Power<base, 0>
			{
				enum { 
					/// <summary>
					/// Contains the power result.
					/// </summary>
					value = 1 
				};
			};


		}
	}
}
