#pragma once
#include "services/messaging/metadata/Metadatum.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			namespace metadata
			{
				/**********************************************************************************************//**
				 * @brief	Stores metadata about the input source file which 
				 * 			was being processed by a parser when a message has
				 * 			been triggered.
				 * 			
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @sa	axis::services::messaging::metadata::Metadatum
				 **************************************************************************************************/
				class AXISSYSTEMBASE_API SourceFileMetadata : public Metadatum
				{
				private:
					 axis::String _sourceFileLocation;
					 unsigned long _lineIndex;
				public:

					/**********************************************************************************************//**
					 * @brief	Creates a new instance of this object.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	sourceFileLocation	A string representing the source file location.
					 * @param	lineIndex		  	Index of the line which was being processed.
					 **************************************************************************************************/
					SourceFileMetadata(const axis::String& sourceFileLocation, unsigned long lineIndex);

					/**********************************************************************************************//**
					 * @brief	Destructor.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					virtual ~SourceFileMetadata(void);

					/**********************************************************************************************//**
					 * @brief	Returns the name of this class.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	A string object containing the class name.
					 **************************************************************************************************/
					static axis::String GetClassName(void);

					/**********************************************************************************************//**
					 * @brief	Makes a deep copy of this object.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	A copy of this object.
					 **************************************************************************************************/
					virtual Metadatum& Clone( void ) const;

					/**********************************************************************************************//**
					 * @brief	Destroys this object.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					virtual void Destroy( void ) const;

					/**********************************************************************************************//**
					 * @brief	Returns the source file location.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	The source file location.
					 **************************************************************************************************/
					axis::String GetSourceFileLocation(void) const;

					/**********************************************************************************************//**
					 * @brief	Gets the line index.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	The line index.
					 **************************************************************************************************/
					unsigned long GetLineIndex(void) const;

					/**********************************************************************************************//**
					 * @brief	Gets the type name for the class of this object.
					 * 			@remark This string will be used to identify instances of
					 * 					this class in metadata collection.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	The type name.
					 **************************************************************************************************/
					virtual axis::String GetTypeName( void ) const;
				};

			}
		}
	}
}

