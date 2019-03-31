#include "EventLogMessageFilter.hpp"

using namespace axis::services::messaging;

axis::services::messaging::filters::EventLogMessageFilter::EventLogMessageFilter( void )
{
	_errorSeverity = axis::services::messaging::ErrorMessage::ErrorLow;
	_warningSeverity = axis::services::messaging::WarningMessage::WarningLow;
	_infoLevel = axis::services::messaging::InfoMessage::InfoNormal;
}

axis::services::messaging::filters::EventLogMessageFilter::EventLogMessageFilter( axis::services::messaging::ErrorMessage::Severity minErrorSeverity )
{
	_errorSeverity = minErrorSeverity;
	_warningSeverity = axis::services::messaging::WarningMessage::WarningLow;
	_infoLevel = axis::services::messaging::InfoMessage::InfoNormal;
}

axis::services::messaging::filters::EventLogMessageFilter::EventLogMessageFilter( axis::services::messaging::WarningMessage::Severity minWarningSeverity )
{
	_errorSeverity = axis::services::messaging::ErrorMessage::ErrorLow;
	_warningSeverity = minWarningSeverity;
	_infoLevel = axis::services::messaging::InfoMessage::InfoNormal;
}

axis::services::messaging::filters::EventLogMessageFilter::EventLogMessageFilter( axis::services::messaging::ErrorMessage::Severity minErrorSeverity, axis::services::messaging::WarningMessage::Severity minWarningSeverity )
{
	_errorSeverity = minErrorSeverity;
	_warningSeverity = minWarningSeverity;
	_infoLevel = axis::services::messaging::InfoMessage::InfoNormal;
}

axis::services::messaging::filters::EventLogMessageFilter::EventLogMessageFilter( axis::services::messaging::InfoMessage::InfoLevel minInfoLevel )
{
	_errorSeverity = axis::services::messaging::ErrorMessage::ErrorLow;
	_warningSeverity = axis::services::messaging::WarningMessage::WarningLow;
	_infoLevel = minInfoLevel;
}

axis::services::messaging::filters::EventLogMessageFilter::EventLogMessageFilter( axis::services::messaging::WarningMessage::Severity minWarningSeverity, axis::services::messaging::InfoMessage::InfoLevel minInfoLevel )
{
	_errorSeverity = axis::services::messaging::ErrorMessage::ErrorLow;
	_warningSeverity = minWarningSeverity;
	_infoLevel = minInfoLevel;
}

axis::services::messaging::filters::EventLogMessageFilter::EventLogMessageFilter( axis::services::messaging::ErrorMessage::Severity minErrorSeverity, axis::services::messaging::WarningMessage::Severity minWarningSeverity, axis::services::messaging::InfoMessage::InfoLevel minInfoLevel )
{
	_errorSeverity = minErrorSeverity;
	_warningSeverity = minWarningSeverity;
	_infoLevel = minInfoLevel;
}

axis::services::messaging::filters::EventLogMessageFilter::~EventLogMessageFilter( void )
{
	// nothing to do here
}

bool axis::services::messaging::filters::EventLogMessageFilter::IsEventMessageFiltered( const axis::services::messaging::EventMessage& message )
{
	// filter event messages only if its severity is lower than we
	// expect
	if (message.IsError())
	{
		const ErrorMessage& errorMsg = static_cast<const ErrorMessage&>(message);
		return errorMsg.GetSeverity() < _errorSeverity;
	}
	else if (message.IsWarning())
	{
		const WarningMessage& warnMsg = static_cast<const WarningMessage&>(message);
		return warnMsg.GetSeverity() < _warningSeverity;
	}
	else if (message.IsInfo())
	{
		const InfoMessage& infoMsg = static_cast<const InfoMessage&>(message);
		return infoMsg.GetInfoLevel() < _infoLevel;
	}

	// a log message
	return false;
}

bool axis::services::messaging::filters::EventLogMessageFilter::IsResultMessageFiltered( const axis::services::messaging::ResultMessage& )
{
	// any result message is filtered
	return true;
}

void axis::services::messaging::filters::EventLogMessageFilter::Destroy( void ) const
{
	delete this;
}

axis::services::messaging::filters::MessageFilter& axis::services::messaging::filters::EventLogMessageFilter::Clone( void ) const
{
	return *new axis::services::messaging::filters::EventLogMessageFilter(*this);
}

axis::services::messaging::ErrorMessage::Severity axis::services::messaging::filters::EventLogMessageFilter::GetMinErrorSeverity( void ) const
{
	return _errorSeverity;
}

axis::services::messaging::WarningMessage::Severity axis::services::messaging::filters::EventLogMessageFilter::GetMinWarningSeverity( void ) const
{
	return _warningSeverity;
}

axis::services::messaging::InfoMessage::InfoLevel axis::services::messaging::filters::EventLogMessageFilter::GetMinInfoLevel( void ) const
{
	return _infoLevel;
}

void axis::services::messaging::filters::EventLogMessageFilter::SetMinErrorSeverity( axis::services::messaging::ErrorMessage::Severity errorSeverity )
{
	_errorSeverity = errorSeverity;
}

void axis::services::messaging::filters::EventLogMessageFilter::SetMinWarningSeverity( axis::services::messaging::WarningMessage::Severity warningSeverity )
{
	_warningSeverity = warningSeverity;
}

void axis::services::messaging::filters::EventLogMessageFilter::SetMinInfoLevel( axis::services::messaging::InfoMessage::InfoLevel infoLevel )
{
	_infoLevel = infoLevel;
}
