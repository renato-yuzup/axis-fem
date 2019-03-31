#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "services/messaging/filters/MessageFilter.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "services/messaging/WarningMessage.hpp"
#include "services/messaging/InfoMessage.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			namespace filters
			{
				/**********************************************************************************************//**
				 * @brief	Implements that only accept a predefined set of
				 * 			event messages.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @sa	MessageFilter
				 **************************************************************************************************/
				class AXISSYSTEMBASE_API EventLogMessageFilter : public MessageFilter
				{
				private:
					axis::services::messaging::ErrorMessage::Severity _errorSeverity;
					axis::services::messaging::WarningMessage::Severity _warningSeverity;
					axis::services::messaging::InfoMessage::InfoLevel _infoLevel;
				public:

					/**********************************************************************************************//**
					 * @brief	Creates a new event message filter.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					EventLogMessageFilter(void);

					/**********************************************************************************************//**
					 * @brief	Creates new event message filter.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	minErrorSeverity	If the received message is an error message, 
					 * 								the minimum error severity which it should have 
					 * 								in order for not being filtered.
					 **************************************************************************************************/
					EventLogMessageFilter(axis::services::messaging::ErrorMessage::Severity minErrorSeverity);

					/**********************************************************************************************//**
					 * @brief	Creates new event message filter.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	minWarningSeverity	If the received message is a warning message,
					 * 								the minimum warning severity which it should have 
					 * 								in order for not being filtered.
					 **************************************************************************************************/
					EventLogMessageFilter(axis::services::messaging::WarningMessage::Severity minWarningSeverity);

					/**********************************************************************************************//**
					 * @brief	Creates new event message filter.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	minInfoLevel		If the received message is an info message,
					 * 								the minimum information level which it should have 
					 * 								in order for not being filtered.
					 **************************************************************************************************/
					EventLogMessageFilter(axis::services::messaging::InfoMessage::InfoLevel minInfoLevel);

					/**********************************************************************************************//**
					 * @brief	Creates new event message filter.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	minErrorSeverity	If the received message is an error message, 
					 * 								the minimum error severity which it should have 
					 * 								in order for not being filtered.
					 * @param	minWarningSeverity	If the received message is a warning message,
					 * 								the minimum warning severity which it should have 
					 * 								in order for not being filtered.
					 **************************************************************************************************/
					EventLogMessageFilter(axis::services::messaging::ErrorMessage::Severity minErrorSeverity, axis::services::messaging::WarningMessage::Severity minWarningSeverity);

					/**********************************************************************************************//**
					 * @brief	Creates new event message filter.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	minWarningSeverity	If the received message is a warning message,
					 * 								the minimum warning severity which it should have 
					 * 								in order for not being filtered.
					 * @param	minInfoLevel		If the received message is an info message,
					 * 								the minimum information level which it should have 
					 * 								in order for not being filtered.
					 **************************************************************************************************/
					EventLogMessageFilter(axis::services::messaging::WarningMessage::Severity minWarningSeverity, axis::services::messaging::InfoMessage::InfoLevel minInfoLevel);

					/**********************************************************************************************//**
					 * @brief	Creates new event message filter.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	minErrorSeverity	If the received message is an error message, 
					 * 								the minimum error severity which it should have 
					 * 								in order for not being filtered.
					 * @param	minWarningSeverity	If the received message is a warning message,
					 * 								the minimum warning severity which it should have 
					 * 								in order for not being filtered.
					 * @param	minInfoLevel		If the received message is an info message,
					 * 								the minimum information level which it should have 
					 * 								in order for not being filtered.
					 **************************************************************************************************/
					EventLogMessageFilter(axis::services::messaging::ErrorMessage::Severity minErrorSeverity, axis::services::messaging::WarningMessage::Severity minWarningSeverity, axis::services::messaging::InfoMessage::InfoLevel mininfoLevel);

					/**********************************************************************************************//**
					 * @brief	Destructor.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					~EventLogMessageFilter(void);

					/**********************************************************************************************//**
					 * @brief	Queries if an event message should be filtered.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	message	The message.
					 *
					 * @return	true if the message should be consumed (that is,
					 * 			accepted, but not forwarded for processing), 
					 * 			false otherwise.
					 **************************************************************************************************/
					virtual bool IsEventMessageFiltered( const axis::services::messaging::EventMessage& message );

					/**********************************************************************************************//**
					 * @brief	Queries if a result message should be filtered.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	message	The message.
					 *
					 * @return	true if the message should be consumed (that is,
					 * 			accepted, but not forwarded for processing), 
					 * 			false otherwise.
					 **************************************************************************************************/
					virtual bool IsResultMessageFiltered( const axis::services::messaging::ResultMessage& message );

					/**********************************************************************************************//**
					 * @brief	Destroys this object.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					virtual void Destroy( void ) const;

					/**********************************************************************************************//**
					 * @brief	Makes a deep copy of this object.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	A copy of this object.
					 **************************************************************************************************/
					virtual MessageFilter& Clone( void ) const;

					/**********************************************************************************************//**
					 * @brief	Returns the minimum error severity for accepted 
					 * 			error messages by this filter.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	The minimum error severity.
					 **************************************************************************************************/
					axis::services::messaging::ErrorMessage::Severity GetMinErrorSeverity(void) const;

					/**********************************************************************************************//**
					 * @brief	Returns the minimum warning severity for accepted 
					 * 			warning messages by this filter.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	The minimum warning severity.
					 **************************************************************************************************/
					axis::services::messaging::WarningMessage::Severity GetMinWarningSeverity(void) const;

					/**********************************************************************************************//**
					 * @brief	Returns the minimum information level for accepted 
					 * 			info messages by this filter.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	The minimum information level.
					 **************************************************************************************************/
					axis::services::messaging::InfoMessage::InfoLevel GetMinInfoLevel(void) const;

					/**********************************************************************************************//**
					 * @brief	Sets the minimum error severity for error messages 
					 * 			accepted by this filter.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	errorSeverity	The error severity.
					 **************************************************************************************************/
					void SetMinErrorSeverity(axis::services::messaging::ErrorMessage::Severity errorSeverity);

					/**********************************************************************************************//**
					 * @brief	Sets the minimum warning severity for warning messages 
					 * 			accepted by this filter.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	warningSeverity	The warning severity.
					 **************************************************************************************************/
					void SetMinWarningSeverity(axis::services::messaging::WarningMessage::Severity warningSeverity);

					/**********************************************************************************************//**
					 * @brief	Sets the minimum information level for info messages 
					 * 			accepted by this filter.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	infoLevel	The information level.
					 **************************************************************************************************/
					void SetMinInfoLevel(axis::services::messaging::InfoMessage::InfoLevel infoLevel);
				};			
			}
		}
	}
}

