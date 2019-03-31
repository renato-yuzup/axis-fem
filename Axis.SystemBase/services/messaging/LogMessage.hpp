#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "services/messaging/EventMessage.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			/**********************************************************************************************//**
			 * @brief	Declares an event message that is normally written to 
			 * 			the log output.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	24 ago 2012
			 *
			 * @sa	EventMessage
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API LogMessage : public EventMessage
			{
			public:

				/**********************************************************************************************//**
				 * @brief	Values that indicates log section opening behavior.
				 **************************************************************************************************/
				enum SectionIndicator
				{
					///< Open a section.
					SectionOpen = 1,
					///< Close a section.
					SectionClose = 2
				};

				/**********************************************************************************************//**
				 * @brief	Values that indicates log section nesting behavior.
				 **************************************************************************************************/
				enum NestingIndicator
				{
					///< Open a new nesting level.
					NestingOpen = 1,
					///< Close current nesting level.
					NestingClose = 2
				};

				/**********************************************************************************************//**
				 * @brief	Values that indicates log block separation behavior.
				 **************************************************************************************************/
				enum BlockIndicator
				{
					///< Open a new block.
					BlockOpen = 1,
					///< Close current block.
					BlockClose = 2
				};

				/**********************************************************************************************//**
				 * @brief	Values that indicates banner printing behavior.
				 **************************************************************************************************/
				enum BannerIndicator
				{
					///< Print splash (heading) banner.
					BannerStart = 1,
					///< Print end (footer) banner.
					BannerEnd = 2
				};
			private:
				int _sectionState;
				int _nestingState;
				int _blockState;
				int _bannerState;
			public:
				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	  The message identifier.
         * @param scope (Optional) Specifies the scope of this message, which influence how it will be
         *              filtered.
				 **************************************************************************************************/

        /**
         * Constructor.
         *
         * @param message The message.
         */
				LogMessage(const axis::String& message);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	   	The message identifier.
				 * @param	message	A description string carried by the message.
				 * @param	level	  The informational level of the message.
				 **************************************************************************************************/
				LogMessage(const axis::String& message, const axis::String& title);

				/**********************************************************************************************//**
				 * @brief	Creates a new log message that indicates to 
				 * 			open/close a log section.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	sectionState	Indicates whether should a section be opened or closed.
				 * @param	bannerString	The section name; relevant only when opening a section.
				 **************************************************************************************************/
				LogMessage(SectionIndicator sectionState, const axis::String& bannerString);

				/**********************************************************************************************//**
				 * @brief	Creates a new log message that indicates to start/close
				 * 			a nesting level.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	nestingState	Tells whether intenting should be 
				 * 							        increased or decreased.
				 **************************************************************************************************/
				explicit LogMessage(NestingIndicator nestingState);

				/**********************************************************************************************//**
				 * @brief	Creates a new log message that indicates to 
				 * 			open/close a block.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	blockState	Tells whether to open or close a block.
				 **************************************************************************************************/
				explicit LogMessage(BlockIndicator blockState);

				/**********************************************************************************************//**
				 * @brief	Creates a new log message that indicates to
				 * 			print an informational banner.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	bannerState	Tells which banner to print.
				 **************************************************************************************************/
				explicit LogMessage(BannerIndicator bannerState);

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				virtual ~LogMessage(void);

				/**********************************************************************************************//**
				 * @brief	Queries if this object is an error message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is an error message, false if otherwise.
				 **************************************************************************************************/
				virtual bool IsError( void ) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this object is a warning message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is a warning message, false if otherwise.
				 **************************************************************************************************/
				virtual bool IsWarning( void ) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this object is an informational message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is an informational message, false if otherwise.
				 **************************************************************************************************/
				virtual bool IsInfo( void ) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this object is a log message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is a log message, false if otherwise.
				 **************************************************************************************************/
				virtual bool IsLogEntry( void ) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this object carries a log action instead of
				 * 			a descriptive message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is a log action, false otherwise.
				 **************************************************************************************************/
				bool IsLogCommand(void) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this message is a log action requesting to
				 * 			open a new section.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is such a message, false otherwise.
				 **************************************************************************************************/
				bool DoesStartNewSection(void) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this message is a log action requesting to
				 * 			increase ident.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is such a message, false otherwise.
				 **************************************************************************************************/
				bool DoesStartNewNesting(void) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this message is a log action requesting to
				 * 			start a new block.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is such a message, false otherwise.
				 **************************************************************************************************/
				bool DoesStartNewBlock(void) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this message is a log action requesting to
				 * 			print a splash banner.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is such a message, false otherwise.
				 **************************************************************************************************/
				bool DoesStartNewBanner(void) const;


				/**********************************************************************************************//**
				 * @brief	Queries if this message is a log action requesting to
				 * 			close a section.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is such a message, false otherwise.
				 **************************************************************************************************/
				bool DoesCloseSection(void) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this message is a log action requesting to
				 * 			decrease indent.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is such a message, false otherwise.
				 **************************************************************************************************/
				bool DoesCloseNesting(void) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this message is a log action requesting to
				 * 			close block.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is such a message, false otherwise.
				 **************************************************************************************************/
				bool DoesCloseBlock(void) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this message is a log action requesting to
				 * 			print an end banner.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is such a message, false otherwise.
				 **************************************************************************************************/
				bool DoesCloseBanner(void) const;
			protected:
				/**********************************************************************************************//**
				 * @brief	Clears resources used by this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				virtual void DoDestroy( void ) const;

				/**********************************************************************************************//**
				 * @brief	Creates a copy this object and its specific properties.
				 * 			@remark	This method is called by the method Clone() of
				 * 					base class Message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	The message identifier.
				 *
				 * @return	A reference to new object.
				 **************************************************************************************************/
				virtual Message& CloneMyself( id_type id ) const;
			};
		}
	}
}

