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
			 * Provides services for string collation according to a locale.
			 */
			class AXISLOCALE_API CollationFacet
			{
			protected:
				enum DefaultCollationType
				{
					Identical,
					Strict,
					IgnoreCase,
					IgnoreCaseAndAccents
				};
			private:
				CollationFacet(const CollationFacet& other);
				CollationFacet& operator =(const CollationFacet& other);

				class CollationData;
				CollationData *_data;

				/**
				 * Effectively compares two strings in strict mode.
				 *
				 * @param s1 The first string.
				 * @param s2 The second string.
				 *
				 * @return A value less than zero if the first string comes before the other,
				 * 		   the null value (zero) if both strings are the same and a value
				 * 		   greater than zero if the second string comes before the other.
				 */
				virtual int DoCompareStrict(const axis::String& s1, const axis::String& s2) const;

				/**
				 * Effectively compares two strings ignoring case.
				 *
				 * @param s1 The first string.
				 * @param s2 The second string.
				 *
				 * @return A value less than zero if the first string comes before the other,
				 * 		   the null value (zero) if both strings are the same and a value
				 * 		   greater than zero if the second string comes before the other.
				 */
				virtual int DoCompareIgnoreCase(const axis::String& s1, const axis::String& s2) const;

				/**
				 * Effectively compares two strings ignoring character case and accents.
				 *
				 * @param s1 The first string.
				 * @param s2 The second string.
				 *
				 * @return A value less than zero if the first string comes before the other,
				 * 		   the null value (zero) if both strings are the same and a value
				 * 		   greater than zero if the second string comes before the other.
				 */
				virtual int DoCompareIgnoreCaseAndAccents(const axis::String& s1, const axis::String& s2) const;

				/**
				 * Tells which is the default string collation for this locale.
				 *
				 * @return The default collation type.
				 */
				virtual DefaultCollationType GetDefaultCollation(void) const;
			public:

				/**
				 * Default constructor.
				 */
				CollationFacet(void);

				/**
				 * Destructor.
				 */
				virtual ~CollationFacet(void);

				/**
				 * Compares two strings in terms of code point.
				 *
				 * @param s1 The first string.
				 * @param s2 The second string.
				 *
				 * @return A value less than zero if the first string comes before the other,
				 * 		   the null value (zero) if both strings are the same and a value
				 * 		   greater than zero if the second string comes before the other.
				 */
				int CompareIdentical(const axis::String& s1, const axis::String& s2) const;

				/**
				 * Compares two strings in terms of character sequence,
				 * but not considering code points.
				 *
				 * @param s1 The first string.
				 * @param s2 The second string.
				 *
				 * @return A value less than zero if the first string comes before the other,
				 * 		   the null value (zero) if both strings are the same and a value
				 * 		   greater than zero if the second string comes before the other.
				 */
				int CompareStrict(const axis::String& s1, const axis::String& s2) const;

				/**
				 * Compares two strings ignoring the character case.
				 *
				 * @param s1 The first string.
				 * @param s2 The second string.
				 *
				 * @return A value less than zero if the first string comes before the other,
				 * 		   the null value (zero) if both strings are the same and a value
				 * 		   greater than zero if the second string comes before the other.
				 */
				int CompareIgnoreCase(const axis::String& s1, const axis::String& s2) const;

				/**
				 * Compares two strings ignoring character case and accents.
				 *
				 * @param s1 The first string.
				 * @param s2 The second string.
				 *
				 * @return A value less than zero if the first string comes before the other,
				 * 		   the null value (zero) if both strings are the same and a value
				 * 		   greater than zero if the second string comes before the other.
				 */
				int CompareIgnoreCaseAndAccents(const axis::String& s1, const axis::String& s2) const;

				/**
				 * Compares two strings using the preferable collation method for
				 * the locale this facet is associated to.
				 *
				 * @param s1 The first string.
				 * @param s2 The second string.
				 *
				 * @return A value less than zero if the first string comes before the other,
				 * 		   the null value (zero) if both strings are the same and a value
				 * 		   greater than zero if the second string comes before the other.
				 */
				int Compare(const axis::String& s1, const axis::String& s2) const;
			};
		}
	}
}

