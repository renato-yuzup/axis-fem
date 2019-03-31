#pragma once
#include "foundation/Axis.Locale.hpp"
#include "TranslationTome.hpp"
#include "AxisString.hpp"

namespace axis
{
	namespace services
	{
		namespace locales
		{
			/**
			 * Provides translation services for known messages identifiers according
			 * to language locale.
			 */
			class AXISLOCALE_API TranslationFacet
			{
			private:
				class TranslationData;
				TranslationData *_data;

				// disallow copy construction and copy assignment
				TranslationFacet(const TranslationFacet& other);
				TranslationFacet& operator =(const TranslationFacet& other);
			public:

				/**
				 * Default constructor.
				 */
				TranslationFacet(void);

				/**
				 * Destructor.
				 */
				~TranslationFacet(void);

				/**
				 * Translates the given message identifier.
				 *
				 * @param messageId Identifier for the message.
				 *
				 * @return The translated message in the default language of the locale.
				 */
				axis::String Translate(long messageId) const;

				/**
				 * Determines if the specified message identifier is recognized 
				 * for translation by this locale.
				 *
				 * @param messageId Identifier for the message.
				 *
				 * @return true if it can translate, false if not.
				 */
				bool CanTranslate(long messageId) const;

				/**
				 * Adds a translation tome to query identifiers for translation.
				 *
				 * @param tome The tome.
				 */
				void AddTome(const axis::services::locales::TranslationTome& tome);

				/**
				 * Returns how many translation tomes is held for this locale.
				 *
				 * @return The tome count.
				 */
				size_type GetTomeCount(void) const;

				/**
				 * Returns the total number of translation entries in all tomes.
				 *
				 * @return The total entries count.
				 */
				size_type GetTotalEntriesCount(void) const;
			};
		}
	}
}

