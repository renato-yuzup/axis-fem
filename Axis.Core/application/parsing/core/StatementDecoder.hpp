#pragma once

#include <utility>
#include <map>
#include <list>
#include <set>
#include <boost/spirit/include/qi.hpp>
#include "AxisString.hpp"
#include "application/parsing/parsers/BlockParser.hpp"
#include "services/language/syntax/BlockHeaderParser.hpp"
#include "services/language/syntax/BlockTailParser.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "application/parsing/core/ParseContext.hpp"
#include "services/messaging/CollectorHub.hpp"

namespace axis { namespace application { namespace parsing { 

using namespace boost::spirit;

namespace core {

class StatementDecoder : public axis::services::messaging::CollectorHub
{
public:
  enum ErrorState
  {
    NoError = 0,
    ParseError = 1,
    CriticalError = 2
  };
	StatementDecoder(axis::application::parsing::core::ParseContext& parseContext);
	~StatementDecoder(void);

  /**********************************************************************************************//**
    * <summary> Requests processing of a new instruction line.</summary>
    *
    * <returns> true if it succeeds, false if it fails.</returns>
    **************************************************************************************************/
	bool ProcessLine(void);

  /**********************************************************************************************//**
    * <summary> Appends to the end of the decoder buffer more instructions to parse.</summary>
    *
    * <param name="instructionChunk"> The instructions that will be added.</param>
    **************************************************************************************************/
  void FeedDecoderBuffer(const axis::String& instructionChunk);

	/**********************************************************************************************//**
		* @fn	virtual void :::SetCurrentContext(BlockParser& parser) = 0 };
		*
		* @brief	Sets the current parsing context.
		*
		* @author	Renato T. Yamassaki
		* @date	14 mar 2011
		*
		* @param [in,out]	parser	The parser representing the current context.
		**************************************************************************************************/
	void SetCurrentContext(axis::application::parsing::parsers::BlockParser& parser);

	void EndProcessing(void);

	bool IsBufferEmpty( void ) const;

	ErrorState GetErrorState( void ) const;

	axis::String GetBufferContents( void ) const;

	unsigned long GetBufferSize(void) const;

	unsigned long GetMaxBufferLength(void) const;
	void SetMaxBufferLength(unsigned long length);
private:
  enum ParsingState
  {
    Accepted = 0,
    NotAccepted = 1,
    AbortRequest = 2,
    SyntaxError = 3
  };

  typedef std::map<axis::String, axis::String> param_list;
  typedef std::list<axis::String> name_stack;
  typedef std::list<axis::application::parsing::parsers::BlockParser *> context_stack;
  typedef std::set<axis::application::parsing::parsers::BlockParser *> open_context_list;

  void InitGrammar(void);
  void StartBlockProcessing(const axis::String& blockName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList);
  void CloseBlockProcessing(axis::String& blockName);

  ParsingState ForwardBufferParsingTask(const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
  ParsingState TestAndRunForBlockTail(const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );
  ParsingState TestAndRunForBlockHeader(const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );

  void RunRecoverySteps(const axis::services::language::iterators::InputIterator& lastReadPosition);
  void TryRecoverySteps(void);
  bool ValidateInputBuffer(void);
  void ConsumeBuffer(const axis::services::language::iterators::InputIterator& lastReadPos, 
    const axis::services::language::iterators::InputIterator& bufferEndPos);

  void RegisterErrorState(ErrorState state);

  // parsing context which was initially provided for us
  axis::application::parsing::core::ParseContext& _parseContext;

  // Buffer which stores characters left to parse (remember that we accept
  // multi line declarations)
  axis::String _parseBuffer;

  // our most critical error state
  ErrorState _errorState;

  // Flag indicating when we are in the middle of a "compiler" error recovery process
  bool _onErrorRecovery;

  bool _onBlockHeadRead;	// we are reading the block heading
  bool _onBlockTailRead;	// we are reading the block tail
  bool _onBodyRead;		// we are reading the block contents

  bool _wasFinalized;
  unsigned long _maxBufferLength;

  // our grammar rules
  axis::services::language::syntax::BlockHeaderParser _blockHeaderParser;
  axis::services::language::syntax::BlockTailParser _blockTailParser;

  // recorded values from block header processing
  param_list _blockParameters;
  axis::String _nestedBlockName;

  // this tell us which is the current block
  axis::String _currentBlockName;
  axis::application::parsing::parsers::BlockParser *_currentContext;

  // this holds our context history
  context_stack _contextStack;
  open_context_list _openedContexts;
  name_stack _blockNameHistory;
};

} } } } // namespace axis::application::parsing::core
