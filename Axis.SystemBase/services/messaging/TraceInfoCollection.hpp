#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "services/messaging/TraceInfo.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			/**********************************************************************************************//**
			 * @brief	A stack of trace information objects.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	25 ago 2012
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API TraceInfoCollection
			{
			public:

				/**********************************************************************************************//**
				 * @brief	A node (or level) of the trace information stack.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				class AXISSYSTEMBASE_API StackNode
				{
				public:

					/**********************************************************************************************//**
					 * @brief	Defines an alias representing the value type stored in this node.
					 **************************************************************************************************/
					typedef TraceInfo value_type;

					/**********************************************************************************************//**
					 * @brief	Defines an alias representing the type of this object.
					 **************************************************************************************************/
					typedef StackNode self;
				private:
					TraceInfo _traceInfo;
					self *_next;
					self *_previous;
				public:

					/**********************************************************************************************//**
					 * @brief	Creates a new node.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @param	info	The item referenced by this node.
					 **************************************************************************************************/
					StackNode(const TraceInfo& info);

					/**********************************************************************************************//**
					 * @brief	Creates a new node.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @param	info			The item referenced by this node.
					 * @param [in,out]	next	The following node in the stack.
					 **************************************************************************************************/
					StackNode(const TraceInfo& info, self& next);

					/**********************************************************************************************//**
					 * @brief	Creates a new node.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @param	info				The item referenced by this node.
					 * @param [in,out]	next		The next node in the stack.
					 * @param [in,out]	previous	The previous node in the stack.
					 **************************************************************************************************/
					StackNode(const TraceInfo& info, self& next, self& previous);

					/**********************************************************************************************//**
					 * @brief	Assigns a reference (link) to the next node in the stack.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @param [in,out]	next	The next node.
					 **************************************************************************************************/
					void ChainNext(self& next);

					/**********************************************************************************************//**
					 * @brief	Assigns a reference (link) to the previous node in the stack.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @param [in,out]	previous	The previous node.
					 **************************************************************************************************/
					void ChainPrevious(self& previous);

					/**********************************************************************************************//**
					 * @brief	Returns the next node in the stack.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	A pointer to the next node or null if no reference was previously set.
					 **************************************************************************************************/
					const self * Next(void) const;

					/**********************************************************************************************//**
					 * @brief	Returns the next node in the stack.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	A pointer to the next node or null if no reference was previously set.
					 **************************************************************************************************/
					self * Next(void);

					/**********************************************************************************************//**
					 * @brief	Returns the previous node in the stack.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	A pointer to the previous node or null if no reference was previously set.
					 **************************************************************************************************/
					const self * Previous(void) const;

					/**********************************************************************************************//**
					 * @brief	Returns the previous node in the stack.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	A pointer to the previous node or null if no reference was previously set.
					 **************************************************************************************************/
					self * Previous(void);

					/**********************************************************************************************//**
					 * @brief	Returns the trace information referenced by this node.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	A reference to the value object.
					 **************************************************************************************************/
					const value_type& Value(void) const;
				};
			private:
				StackNode *_first;
				StackNode *_last;
				size_type _count;

				/**********************************************************************************************//**
				 * @brief	Copies another object contents into this object (overwrite operation).
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	other	The source object.
				 **************************************************************************************************/
				void Copy(const TraceInfoCollection& other);
			public:

				/**********************************************************************************************//**
				 * @brief	Defines an alias representing type of the value object.
				 **************************************************************************************************/
				typedef TraceInfo value_type;

				/**********************************************************************************************//**
				 * @brief	Defines an alias representing type of the stack node.
				 **************************************************************************************************/
				typedef StackNode node_type;

				/**********************************************************************************************//**
				 * @brief	Creates a new empty stack.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				TraceInfoCollection(void);

				/**********************************************************************************************//**
				 * @brief	Copy constructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	other	The source object.
				 **************************************************************************************************/
				TraceInfoCollection(const TraceInfoCollection& other);

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				~TraceInfoCollection(void);

				/**********************************************************************************************//**
				 * @brief	Copy assignment operator.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	other	The source object.
				 *
				 * @return	A reference to this object.
				 **************************************************************************************************/
				TraceInfoCollection& operator =(const TraceInfoCollection& other);

				/**********************************************************************************************//**
				 * @brief	Adds a copy of the specified trace information into the stack.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	info	The trace information.
				 **************************************************************************************************/
				void AddTraceInfo(const TraceInfo& info);

				/**********************************************************************************************//**
				 * @brief	Returns the top object and removes it from the stack.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	A reference to the top object.
				 **************************************************************************************************/
				value_type PopInfo(void);

				/**********************************************************************************************//**
				 * @brief	Returns a reference to the top object in the stack.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	.
				 **************************************************************************************************/
				const value_type& PeekInfo(void) const;

        /**
         * Query if there are any trace information which matches all specified information.
         *
         * @param sourceId The trace source ID.
         *
         * @return true if it has, false otherwise.
         */
        bool Contains(int sourceId) const;

        /**
         * Query if there are any trace information which matches all specified information.
         *
         * @param sourceId   The trace source ID.
         * @param sourceName Name of the source.
         *
         * @return true if it has, false otherwise.
         */
        bool Contains(int sourceId, const axis::String& sourceName) const;

        /**
         * Query if there are any trace information which matches all specified information.
         *
         * @param sourceName Name of the source.
         *
         * @return true if it has, false otherwise.
         */
        bool Contains(const axis::String& sourceName) const;

				/**********************************************************************************************//**
				 * @brief	Clear the stack and releases all resources used by 
				 * 			stored items.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				void Clear(void);

				/**********************************************************************************************//**
				 * @brief	Returns the number of items stored in the stack.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	A non-negative number representing the item count.
				 **************************************************************************************************/
				size_type Count(void) const;

				/**********************************************************************************************//**
				 * @brief	Queries if the stack has no items.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	true if it is empty, false otherwise.
				 **************************************************************************************************/
				bool Empty(void) const;
			};
		}
	}
}

