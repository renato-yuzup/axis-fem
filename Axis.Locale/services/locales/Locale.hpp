#pragma once
#include "foundation/Axis.Locale.hpp"
#include "AxisString.hpp"
#include "DateTimeFacet.hpp"
#include "NumberFacet.hpp"
#include "CollationFacet.hpp"
#include "TranslationFacet.hpp"

namespace axis
{
	namespace services
	{
		namespace locales
		{
			/**
			 * Provides collation, formatting and translation services according
			 * to a region culture.
			 */
			class AXISLOCALE_API Locale
			{
			private:
				axis::String         _localeCode;
				const DateTimeFacet  *_dateTimeFacet;
				const NumberFacet    *_numberFacet;
				const CollationFacet *_stringFacet;
				const TranslationFacet   *_messageFacet;

				Locale(const Locale& other);
				Locale& operator =(const Locale& other);
			public:

				/**
				 * Constructor.
				 *
				 * @param localeCode	   The locale code which follows POSIX naming convention for locales.
				 * @param dateTimeFacet    The date time facet for this locale.
				 * @param numberFacet	   Number of facets for this locale.
				 * @param collationFacet   The string collation facet for this locale.
				 * @param translationFacet The translation facet for this locale.
				 */
				Locale(const axis::String& localeCode,
					   const DateTimeFacet& dateTimeFacet,
					   const NumberFacet& numberFacet,
					   const CollationFacet& collationFacet,
					   const TranslationFacet& translationFacet);

				/**
				 * Destructor.
				 */
				~Locale(void);

				/**
				 * Returns an object that provides date/time formatting services according to locale culture.
				 *
				 * @return The date/time facet for this locale.
				 */
				const DateTimeFacet& GetDataTimeLocale(void) const;

				/**
				 * Returns an object that provides number formatting services according to locale culture.
				 *
				 * @return The number facet for this locale.
				 */
				const NumberFacet& GetNumberLocale(void) const;

				/**
				 * Returns an object that provides string collation services according to locale culture.
				 *
				 * @return The collation facet for this locale.
				 */
				const CollationFacet& GetCollation(void) const;

				/**
				 * Returns an object that provides translation services according to locale culture.
				 *
				 * @return The translation facet for this locale.
				 */
				const TranslationFacet& GetDictionary(void) const;

				/**
				 * Returns the locale code for this locale.
				 *
				 * @return The locale code string according to POSIX naming convention for locales.
				 */
				axis::String GetLocaleCode(void) const;

				/**
				 * Returns if this locale is equal to another.
				 *
				 * @param other The other locale.
				 *
				 * @return true if both are equal, false otherwise.
				 */
				bool operator ==(const Locale& other) const;

				/**
				 * Returns if this locale is different from another.
				 *
				 * @param other The other locale.
				 *
				 * @return true if both are different, false otherwise.
				 */
				bool operator !=(const Locale& other) const;
			};
		}
	}
}

