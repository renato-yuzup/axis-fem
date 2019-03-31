#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "AxisString.hpp"
#include "services/messaging/TraceTag.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			/**********************************************************************************************//**
			 * @brief	A tag object which retains information about a hop object 
			 * 			from which a message has been forwarded and optionally
			 * 			additional information about the state of that object at 
			 * 			that time.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	25 ago 2012
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API TraceInfo
			{
			private:
				axis::String _sourceName;
				int _sourceId;
				TraceTag *_sourceTag;

				/**********************************************************************************************//**
				 * @brief	Copies information from another object into this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	other	Source object.
				 **************************************************************************************************/
				void Copy(const TraceInfo& other);
			public:

				/**********************************************************************************************//**
				 * @brief	Creates a new trace information.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	sourceId	An identifier for the source which stamped
				 * 						this trace information.
				 **************************************************************************************************/
				TraceInfo(int sourceId);

				/**********************************************************************************************//**
				 * @brief	Creates a new trace information.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	sourceName	Name of the source which stamped this 
				 * 						trace information.
				 **************************************************************************************************/
				TraceInfo(const axis::String& sourceName);

				/**********************************************************************************************//**
				 * @brief	Constructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	sourceId	An identifier for the source which stamped
				 * 						this trace information.
				 * @param	sourceName	Name of the source which stamped this 
				 * 						trace information.
				 **************************************************************************************************/
				TraceInfo(int sourceId, const axis::String& sourceName);

				/**********************************************************************************************//**
				 * @brief	Constructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	sourceId	An identifier for the source which stamped
				 * 						this trace information.
				 * @param	sourceName	Name of the source which stamped this 
				 * 						trace information.
				 * @param	tag		  	A tag object for additional information.
				 **************************************************************************************************/
				TraceInfo(int sourceId, const axis::String& sourceName, const TraceTag& tag);

				/**********************************************************************************************//**
				 * @brief	Constructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	sourceId	An identifier for the source which stamped
				 * 						this trace information.
				 * @param	tag		  	A tag object for additional information.
				 **************************************************************************************************/
				TraceInfo(int sourceId, const TraceTag& tag);

				/**********************************************************************************************//**
				 * @brief	Copy constructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	other	The source object.
				 **************************************************************************************************/
				TraceInfo(const TraceInfo& other);

				/**********************************************************************************************//**
				 * @brief	Copy assignment operator.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	other	The other object.
				 *
				 * @return	A reference to this object.
				 **************************************************************************************************/
				TraceInfo& operator =(const TraceInfo& other);

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				~TraceInfo(void);

				/**********************************************************************************************//**
				 * @brief	Returns the source name of the object which stamped
				 * 			this information into the current message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	A string containing the source name. If it was not 
				 * 			specified, an empty string is returned.
				 **************************************************************************************************/
				axis::String& SourceName(void);

				/**********************************************************************************************//**
				 * @brief	Returns the source name of the object which stamped
				 * 			this information into the current message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	A string containing the source name. If it was not 
				 * 			specified, an empty string is returned.
				 **************************************************************************************************/
				const axis::String& SourceName(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the source identifier of the object which stamped
				 * 			this information into the current message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	A non-zero numerical value representing the source 
				 * 			identifier or zero if it was not specified.
				 **************************************************************************************************/
				int& SourceId(void);

				/**********************************************************************************************//**
				 * @brief	Returns the source identifier of the object which stamped
				 * 			this information into the current message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	A non-zero numerical value representing the source 
				 * 			identifier or zero if it was not specified.
				 **************************************************************************************************/
				const int& SourceId(void) const;

				/**********************************************************************************************//**
				 * @brief	Appends a tag into this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	tag	The tag object.
				 **************************************************************************************************/
				void AppendTag(const TraceTag& tag);

				/**********************************************************************************************//**
				 * @brief	Removes current tag and replaces it with another one.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	tag	The new tag object.
				 **************************************************************************************************/
				void ReplaceTag(const TraceTag& tag);

				/**********************************************************************************************//**
				 * @brief	Removes, if any, current tag object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				void EraseTag(void);

				/**********************************************************************************************//**
				 * @brief	Queries if this object has an attached tag on it.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	true if tagged, false otherwise.
				 **************************************************************************************************/
				bool IsTagged(void) const;
			};
		}
	}
}

