#pragma once
#include "foundation/Axis.Locale.hpp"
#include "AxisString.hpp"

namespace axis
{
	namespace services
	{
		namespace locales
		{
			/**
			 * Provides number formatting services according to a specific locale.
			 */
			class AXISLOCALE_API NumberFacet
			{
			private:
				NumberFacet(const NumberFacet& other);
				NumberFacet& operator =(const NumberFacet& other);

				/**
				 * Method that effectively returns a string containing the number formatted as percentage.
				 *
				 * @param percent		 The number.
				 * @param numberOfDigits (optional) number of decimal digits to consider.
				 *
				 * @return The string containing the formatted number.
				 */
				virtual axis::String DoToPercent(double percent, unsigned int numberOfDigits) const;

				/**
				 * Method that effectively returns the string representation of a given number.
				 *
				 * @param number The number.
				 *
				 * @return The string representation of the number.
				 */
				virtual axis::String DoToString(double number) const;

				/**
				 * Method that effectively returns the string representation of a given number.
				 *
				 * @param number		 The number.
				 * @param numberOfDigits Number of decimal digits to consider.
				 *
				 * @return The string representation of the number.
				 */
				virtual axis::String DoToString(double number, unsigned int numberOfDigits) const;

				/**
				 * Method that effectively returns the string representation of a given number.
				 *
				 * @param number The number.
				 *
				 * @return The string representation of the number.
				 */
				virtual axis::String DoToString(long number) const ;

				/**
				 * Method that effectively returns the string representation of a given number.
				 *
				 * @param number The number.
				 *
				 * @return The string representation of the number.
				 */
				virtual axis::String DoToString(unsigned long number) const;

				/**
				 * Method that effectively converts a number to scientific notation.
				 *
				 * @param number The number.
				 *
				 * @return The scientific notation of the given number.
				 */
				virtual axis::String DoToScientificNotation(double number) const;

				/**
				 * Method that effectively converts a number to scientific notation.
				 *
				 * @param number		   The number.
				 * @param maxDecimalDigits The maximum number of decimal digits.
				 *
				 * @return The scientific notation of the given number.
				 */
				virtual axis::String DoToScientificNotation(double number, int maxDecimalDigits) const;

				/**
				 * Method that effectively converts a number to engineering notation.
				 *
				 * @param number The number.
				 *
				 * @return The engineering notation of the given number.
				 */
				virtual axis::String DoToEngineeringNotation(double number) const;

				/**
				 * Method that effectively converts a number to engineering notation.
				 *
				 * @param number		   The number.
				 * @param maxDecimalDigits The maximum number of decimal digits.
				 *
				 * @return The engineering notation of the given number.
				 */
				virtual axis::String DoToEngineeringNotation(double number, int maxDecimalDigits) const;
			public:

				/**
				 * Default constructor.
				 */
				NumberFacet(void);

				/**
				 * Destructor.
				 */
				virtual ~NumberFacet(void);

				/**
				 * Returns a string containing the number formatted as percentage.
				 *
				 * @param percent		 The number.
				 * @param numberOfDigits (optional) number of decimal digits to consider.
				 *
				 * @return The string containing the formatted number.
				 */
				axis::String ToPercent(double percent, unsigned int numberOfDigits = 0) const;

				/**
				 * Returns the string representation of a given number.
				 *
				 * @param number The number.
				 *
				 * @return The string representation of the number.
				 */
				axis::String ToString(double number) const;

				/**
				 * Returns the string representation of a given number.
				 *
				 * @param number		 The number.
				 * @param numberOfDigits Number of decimal digits to consider.
				 *
				 * @return The string representation of the number.
				 */
				axis::String ToString(double number, unsigned int numberOfDigits) const;

				/**
				 * Returns the string representation of a given number.
				 *
				 * @param number The number.
				 *
				 * @return The string representation of the number.
				 */
				axis::String ToString(long number) const;

				/**
				 * Returns the string representation of a given number.
				 *
				 * @param number The number.
				 *
				 * @return The string representation of the number.
				 */
				axis::String ToString(unsigned long number) const;

				/**
				 * Converts a number to scientific notation.
				 *
				 * @param number The number.
				 *
				 * @return The scientific notation of the given number.
				 */
				axis::String ToScientificNotation(double number) const;

				/**
				 * Converts a number to scientific notation.
				 *
				 * @param number		   The number.
				 * @param maxDecimalDigits The maximum number of decimal digits.
				 *
				 * @return The scientific notation of the given number.
				 */
				axis::String ToScientificNotation(double number, int maxDecimalDigits) const;

				/**
				 * Converts a number to engineering notation.
				 *
				 * @param number The number.
				 *
				 * @return The engineering notation of the given number.
				 */
				axis::String ToEngineeringNotation(double number) const;

				/**
				 * Converts a number to engineering notation.
				 *
				 * @param number		   The number.
				 * @param maxDecimalDigits The maximum number of decimal digits.
				 *
				 * @return The engineering notation of the given number.
				 */
				axis::String ToEngineeringNotation(double number, int maxDecimalDigits) const;
			};
		}
	}
}

