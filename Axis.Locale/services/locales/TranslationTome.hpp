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
			 * Retains translation entries for message identifiers.
			 */
			class AXISLOCALE_API TranslationTome
			{
			private:
				// disallow copy construction and copy assignment
				TranslationTome(const TranslationTome& other);
				TranslationTome& operator =(const TranslationTome& other);

				/**
				 * Returns the minor message identifier number in the message range that this tome can translate.
				 *
				 * @return The minor message identifier.
				 */
				virtual long DoGetFirstTranslationIdRange(void) const = 0;

				/**
				 * Returns the major message identifier number in the message range that this tome can translate.
				 *
				 * @return The major message identifier.
				 */
				virtual long DoGetLastTranslationIdRange(void) const = 0;

				/**
				 * Method that effectively translates a given message identifier.
				 *
				 * @param messageId Identifier for the message.
				 *
				 * @return The translated message.
				 */
				virtual axis::String DoTranslate(long messageId) const = 0;

				/**
				 * Method that effectively determines if this tome knows the translation for a given message identifier.
				 *
				 * @param messageId Identifier for the message.
				 *
				 * @return true if we it can translate, false otherwise.
				 */
				virtual bool DoCanTranslate(long messageId) const = 0;

				/**
				 * Method that effectively returns the translation entry count.
				 *
				 * @return The number of translation entries.
				 */
				virtual size_type DoGetEntryCount(void) const = 0;
			public:

				/**
				 * Destructor.
				 */
				virtual ~TranslationTome(void);

				/**
				 * Translates a given message identifier.
				 *
				 * @param messageId Identifier for the message.
				 *
				 * @return The translated message.
				 */
				axis::String Translate(long messageId) const;

				/**
				 * Determine if this tome knows the translation for a given message identifier.
				 *
				 * @param messageId Identifier for the message.
				 *
				 * @return true if we it can translate, false otherwise.
				 */
				bool CanTranslate(long messageId) const;

				/**
				 * Returns the translation entry count.
				 *
				 * @return The number of translation entries.
				 */
				size_type GetEntryCount(void) const;


			};
		}
	}
}

