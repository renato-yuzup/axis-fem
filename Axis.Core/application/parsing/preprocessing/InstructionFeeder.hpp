#pragma once
#include <boost/spirit/include/qi.hpp>
#include "services/io/StreamReader.hpp"
#include "AxisString.hpp"
#include "application/parsing/core/ParseContext.hpp"

namespace axis { namespace application { namespace parsing { namespace preprocessing {

/// <summary>
/// Reads data from an input stream and feeds instructions to the preprocessor,
/// removing adornments such as blank lines and comments from the input.
/// </summary>
class InstructionFeeder
{
private:
	axis::services::io::StreamReader *_source;		// our source stream
	axis::application::parsing::core::ParseContext& _parseContext;

	// We alway look ahead when reading so that we can predict 
	// adornments that were put at the end of the file. Doing
	// this way, we can promptly say when we reached EOF. For us,
	// EOF occurs when no usable line can be found in the stream.
	axis::String _lastLineRead;		// contents of the last line read
	bool _eofFlag;								// is there any more usable lines?
					
	// In order to detect open comment blocks errors, we must
	// signal when we had encountered an open block delimiter.
	bool _isReadingComment;

	// Used when a dangling open comment block delimiter is found.
	// An exception is thrown and the location of the delimiter
	// is told by the contents stored in this attribute.
	unsigned long _blockStartingLine;

	unsigned long _lastLineNumber;
public:
	/// <summary>
	/// Creates a new instance of this class.
	/// </summary>
	InstructionFeeder(axis::application::parsing::core::ParseContext& context);

	/// <summary>
	/// Creates a new instance of this class.
	/// </summary>
	/// <param name="source">Stream from which data shall be extracted.</param>
	InstructionFeeder(axis::services::io::StreamReader& source, 
    axis::application::parsing::core::ParseContext& context);

	/// <summary>
	/// Destroys this object.
	/// </summary>
	virtual ~InstructionFeeder(void);

	/// <summary>
	/// Returns if there is any usable expressions left in the stream.
	/// </summary>
	bool IsEOF(void) const;

	/// <summary>
	/// Read the next expression in the stream.
	/// </summary>
	/// <param name="line">Reference to which extracted expression is put.</param>
	void ReadLine(axis::String& line);

	/// <summary>
	/// Returns the current stream used by this object.
	/// </summary>
	axis::services::io::StreamReader& GetCurrentSource(void) const;

	/// <summary>
	/// Changes the stream used by this object. EOF state is reevaluated
	/// in order to reflect the new stream state.
	/// </summary>
	/// <param name="reader">The new stream to use.</param>
	void ChangeSource(axis::services::io::StreamReader& reader);

	/// <summary>
	/// Pushes back the last line read from the stream so that
	/// it seems like the last line was never read.
	/// </summary>
	void Rewind(void);

	unsigned long GetLastLineReadIndex(void) const;

	/**************************************************************************************************
		* <summary>	Resets the state of this object. </summary>
		**************************************************************************************************/
	void Reset(void);
private:
	/// <summary>
	/// Searches for the next following expression in the stream.
	/// </summary>
	void ReadNextRelevantLine(void);

	/// <summary>
	/// Checks if the line has an expression. If it has, comments and
	/// whitespaces are removed.
	/// </summary>
	/// <param name="line">String to be analysed.</param>
	/// <returns>
	/// True if an expression could be extracted, False otherwise.
	/// </returns>
	bool ParseLine(axis::String& line);
};

} } } } // namespace axis::application::parsing::preprocessing
