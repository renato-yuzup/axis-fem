#include "LogMessage.hpp"

namespace asmm = axis::services::messaging;

asmm::LogMessage::LogMessage( const axis::String& message ) :
EventMessage(0, message)
{
	_sectionState = 0; _nestingState = 0; _blockState = 0; _bannerState = 0;
}

asmm::LogMessage::LogMessage( const axis::String& message, const axis::String& title ) :
EventMessage(0, message, title)
{
	_sectionState = 0; _nestingState = 0; _blockState = 0; _bannerState = 0;
}

asmm::LogMessage::LogMessage( SectionIndicator sectionState, const axis::String& bannerString ) :
EventMessage(0, bannerString)
{
	_sectionState = (int)sectionState;
	_nestingState = 0;
	_blockState = 0;
	_bannerState = 0;
}

asmm::LogMessage::LogMessage( NestingIndicator nestingState ) :
EventMessage(0)
{
	_sectionState = 0;
	_nestingState = (int)nestingState;
	_blockState = 0;
	_bannerState = 0;
}

asmm::LogMessage::LogMessage( BlockIndicator blockState ) :
EventMessage(0)
{
	_sectionState = 0;
	_nestingState = 0;
	_blockState = blockState;
	_bannerState = 0;
}

asmm::LogMessage::LogMessage( BannerIndicator bannerState ) :
EventMessage(0)
{
	_sectionState = 0;
	_nestingState = 0;
	_blockState = 0;
	_bannerState = bannerState;
}

asmm::LogMessage::~LogMessage( void )
{
	// nothing to do
}

bool asmm::LogMessage::IsError( void ) const
{
	return false;
}

bool asmm::LogMessage::IsWarning( void ) const
{
	return false;
}

bool asmm::LogMessage::IsInfo( void ) const
{
	return false;
}

bool asmm::LogMessage::IsLogEntry( void ) const
{
	return true;
}

void asmm::LogMessage::DoDestroy( void ) const
{
	delete this;
}

asmm::Message& asmm::LogMessage::CloneMyself( id_type id ) const
{
	return *new LogMessage(GetDescription(), GetTitle());
}

bool asmm::LogMessage::IsLogCommand( void ) const
{
	return _blockState != 0 || _nestingState != 0 || _sectionState != 0 || _bannerState != 0;
}

bool asmm::LogMessage::DoesStartNewSection( void ) const
{
	return _sectionState == (int)SectionOpen;
}

bool asmm::LogMessage::DoesStartNewNesting( void ) const
{
	return _nestingState == (int)NestingOpen;
}

bool asmm::LogMessage::DoesCloseSection( void ) const
{
	return _sectionState == (int)SectionClose;
}

bool asmm::LogMessage::DoesCloseNesting( void ) const
{
	return _nestingState == (int)NestingClose;
}

bool asmm::LogMessage::DoesStartNewBlock( void ) const
{
	return _blockState == (int)BlockOpen;
}

bool asmm::LogMessage::DoesCloseBlock( void ) const
{
	return _blockState == (int)BlockClose;
}

bool asmm::LogMessage::DoesStartNewBanner( void ) const
{
	return _bannerState == (int)BannerStart;
}

bool asmm::LogMessage::DoesCloseBanner( void ) const
{
	return _bannerState == (int)BannerEnd;
}
