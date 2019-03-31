#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/date_time/Timestamp.hpp"
#include "services/messaging/TraceInfoCollection.hpp"
#include "services/messaging/metadata/MetadataCollection.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			/**********************************************************************************************//**
			 * @brief	Represents any message transmitted through the messaging
			 * 			system of the program.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	24 ago 2012
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API Message
			{
			public:

				/**********************************************************************************************//**
				 * @brief	Defines an alias for the message identifier type.
				 **************************************************************************************************/
				typedef long id_type;

				/**********************************************************************************************//**
				 * @brief	Defines an alias representing this type.
				 **************************************************************************************************/
				typedef Message self;
			private:
				TraceInfoCollection _traceInformation;
				axis::services::messaging::metadata::MetadataCollection *_metadata;

				axis::foundation::date_time::Timestamp _timestamp;
				id_type _id;

				// prohibit use of copy constructor and copy assignment
				Message(const Message& message);
				Message& operator =(const Message& message);
			protected:

				/**********************************************************************************************//**
				 * @brief	Clears resources used by this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				virtual void DoDestroy(void) const = 0;

				/**********************************************************************************************//**
				 * @brief	Executes the clone operation.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	The message identifier.
				 *
				 * @return	A reference to the new object.
				 **************************************************************************************************/
				virtual Message& DoClone(id_type id) const = 0;
			public:

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	The message identifier.
				 **************************************************************************************************/
				Message(id_type id);

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				virtual ~Message(void);

				/**********************************************************************************************//**
				 * @brief	Returns the message identifier.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	The message identifier.
				 **************************************************************************************************/
				id_type GetId(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the creation time for this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	An object containing the timestamp of the message.
				 **************************************************************************************************/
				axis::foundation::date_time::Timestamp GetTimestamp(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns a reference to the collection of trace 
				 * 			objects attached to this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	The trace collection object.
				 **************************************************************************************************/
				const TraceInfoCollection& GetTraceInformation(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns a reference to the collection of trace 
				 * 			objects attached to this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	The trace collection object.
				 **************************************************************************************************/
				TraceInfoCollection& GetTraceInformation(void);

				/**********************************************************************************************//**
				 * @brief	Returns a reference to the collection of metadata
				 * 			objects attached to this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	The metadata collection object.
				 **************************************************************************************************/
				const axis::services::messaging::metadata::MetadataCollection& GetMetadata(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns a reference to the collection of metadata
				 * 			objects attached to this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	The metadata collection object.
				 **************************************************************************************************/
				axis::services::messaging::metadata::MetadataCollection& GetMetadata(void);

				/**********************************************************************************************//**
				 * @brief	Attaches a trace information object to this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	traceInfo	The trace information object.
				 *
				 * @return	A reference to this object.
				 **************************************************************************************************/
				self& operator << (const TraceInfo& traceInfo);

				/**********************************************************************************************//**
				 * @brief	Queries if this object is an event message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is an event message, false otherwise.
				 **************************************************************************************************/
				virtual bool IsEvent(void) const = 0;

				/**********************************************************************************************//**
				 * @brief	Queries if this object is a result message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is a result message, false otherwise.
				 **************************************************************************************************/
				virtual bool IsResult(void) const = 0;

				/**********************************************************************************************//**
				 * @brief	Destroys this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				virtual void Destroy(void) const;

				/**********************************************************************************************//**
				 * @brief	Makes a deep copy of this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	A copy of this object.
				 **************************************************************************************************/
				virtual Message& Clone(void) const;
			};
		}
	}
}

