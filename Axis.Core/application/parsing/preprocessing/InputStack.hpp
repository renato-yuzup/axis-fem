/// <summary>
/// Contains definition for the class axis::application::parsing::preprocessing::StreamStack.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once

#include "InputStack.hpp"
#include "services/io/FileReader.hpp"
#include <set>

namespace axis { namespace application { namespace parsing { namespace preprocessing {

/// <summary>
/// Describes an input stack of files stored in disk.
/// </summary>
class InputStack
{
public:
	/// <summary>
	/// Creates a new instance of this class.
	/// </summary>
	InputStack(void);

	/// <summary>
	/// Destroys this object.
	/// </summary>
	~InputStack(void);

	/// <summary>
	/// Returns how much streams can this object store in the stack.
	/// </summary>
	int GetMaximumSize( void ) const;

	/// <summary>
	/// Sets how many streams can this object store in the stack.
	/// </summary>
	void SetMaximumSize( int maxLength );

	/// <summary>
	/// Adds a new stream to the top of the stack.
	/// </summary>
	int AddStream( axis::String streamDescriptor );

	/// <summary>
	/// Adds a new stream to the top of the stack.
	/// </summary>
	int AddStream( axis::services::io::StreamReader& stream );

	/// <summary>
	/// Returns how many streams are stored in the stack.
	/// </summary>
	int Count( void ) const;

	/// <summary>
	/// Returns a reference to the stream stored in the stack at the specified position.
	/// </summary>
	axis::services::io::StreamReader& GetStream( int id ) const;

	/// <summary>
	/// Returns if this object supports storing one more stream in its current state.
	/// </summary>
	bool CanStore( void ) const;

	/// <summary>
	/// Closes and remove the last stream added to the stack.
	/// </summary>
	void CloseTopStream( void );

	/// <summary>
	/// Returns a reference to the stream stored at the top of the stack.
	/// </summary>
	axis::services::io::StreamReader& GetTopStream( void ) const;

	/// <summary>
	/// Closes and remove all streams but the first from the stack.
	/// </summary>
	void CloseNestedStreams(void);
private:          
  axis::services::io::StreamReader **_stack;
  int _topElement;
  int _maxStackLength;
};

} } } } // namespace axis::application::parsing::preprocessing
