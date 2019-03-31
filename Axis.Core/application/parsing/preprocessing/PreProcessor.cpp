#include "PreProcessor.hpp"

#include "foundation/definitions/AxisInputLanguage.hpp"
#include "foundation/StandardTraceHints.hpp"
#include "foundation/InvalidPreProcessorDirectiveException.hpp"
#include "foundation/InvalidIncludeFileException.hpp"
#include "foundation/InputStackFullException.hpp"
#include "foundation/SymbolRedefinedException.hpp"
#include "foundation/InvalidSyntaxException.hpp"
#include "foundation/UnexpectedExpressionException.hpp"
#include "foundation/MissingDelimiterException.hpp"
#include "foundation/IOException.hpp"

#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/parsing/ParseResult.hpp"
#include "services/language/parsing/ExpressionNode.hpp"
#include "services/language/parsing/EnumerationExpression.hpp"
#include "services/language/actions/CallbackAction.hpp"
#include "services/io/FileSystem.hpp"
#include "services/messaging/ErrorMessage.hpp"

#include "application/parsing/error_messages.hpp"

namespace afd  = axis::foundation::definitions;
namespace asio = axis::services::io;
namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace asla = axis::services::language::actions;
namespace asmm = axis::services::messaging;
namespace aapp = axis::application::parsing::preprocessing;
namespace aapc = axis::application::parsing::core;

aapp::PreProcessor::PreProcessor(aapp::InputStack& inputStack, 
                           aapc::ParseContextConcrete& context) :
  _inputStack(inputStack), _parseContext(context), _formatter(context), 
  _symbol_enum_rule(aslf::AxisGrammar::CreateIdParser(), true)
{
	InitGrammar();
	// init existence parser object
	_existenceParser = new ExistenceExpressionParser(_symbolTable);
	_nestedCondCount = 0;
	_nestedSkippedCondCount = 0;
	_ifScanning = false;
	_scanEndIfBlock = false;
	_skipFile = false;
	_fullStop = false;
	_errorState = NoError;
}

aapp::PreProcessor::~PreProcessor(void)
{
	// delete parser
	delete _existenceParser;
}

void aapp::PreProcessor::InitGrammar( void )
{
	// How every preprocessor directive are identified at the beginning of the line?
	_preprocessor_rule        << aslf::AxisGrammar::CreateReservedWordParser(_T("@"));

	// definitions for the define directive
	_alt_define_rule          << aslf::AxisGrammar::CreateReservedWordParser(_T("define")) 
                            << aslf::AxisGrammar::CreateReservedWordParser(_T("def"));
	_define_rule              << _alt_define_rule << _symbol_enum_rule;

	// Definition for the include directive
	_include_rule             <<  aslf::AxisGrammar::CreateReservedWordParser(_T("include")) 
                            << aslf::AxisGrammar::CreateStringParser(false);

	// definitions for the end and skip directives
	_end_rule                 <<  aslf::AxisGrammar::CreateReservedWordParser(_T("end"));
	_skip_rule                << aslf::AxisGrammar::CreateReservedWordParser(_T("skip"));

	// definitions for the if clauses rules
	_clause_rule              << aslf::AxisGrammar::CreateOperatorParser(    
                                afd::AxisInputLanguage::Operators.OpenGroup.GetSyntax()) 
                            << _expr_rule 
                            << aslf::AxisGrammar::CreateOperatorParser(    
                                afd::AxisInputLanguage::Operators.CloseGroup.GetSyntax());
	_expr_rule                << _bin_op_expr_rule << _term_rule;
	_bin_op_expr_rule         << _term_rule << _bin_op_rule << _expr_rule;
	_term_rule                << _group_rule << _un_op_expr_rule << aslf::AxisGrammar::CreateIdParser();
	_group_rule               << aslf::AxisGrammar::CreateOperatorParser(    
                                afd::AxisInputLanguage::Operators.OpenGroup.GetSyntax()) 
                            << _expr_rule 
                            << aslf::AxisGrammar::CreateOperatorParser(    
                                afd::AxisInputLanguage::Operators.CloseGroup.GetSyntax());
	_un_op_expr_rule          << _un_op_rule << _expr_rule;
	_bin_op_rule              << aslf::AxisGrammar::CreateOperatorParser(    
                                afd::AxisInputLanguage::Operators.OperatorAnd.GetSyntax()) 
                            << aslf::AxisGrammar::CreateOperatorParser(    
                                afd::AxisInputLanguage::Operators.OperatorOr.GetSyntax());
	_un_op_rule               << aslf::AxisGrammar::CreateOperatorParser(    
                                afd::AxisInputLanguage::Operators.OperatorNot.GetSyntax());
	_if_clause_rule           << aslf::AxisGrammar::CreateReservedWordParser(_T("if")) 
                            << aslf::AxisGrammar::CreateReservedWordParser(_T("defined")) 
                            << _clause_rule;
	_elseif_clause_rule       << aslf::AxisGrammar::CreateReservedWordParser(_T("else")) 
                            << aslf::AxisGrammar::CreateReservedWordParser(_T("if")) 
                            << aslf::AxisGrammar::CreateReservedWordParser(_T("defined")) 
                            << _clause_rule;
	_else_clause_rule         << aslf::AxisGrammar::CreateReservedWordParser(_T("else"));
	_endif_clause_rule        << aslf::AxisGrammar::CreateReservedWordParser(_T("endif"));
	_ifnot_clause_rule        << aslf::AxisGrammar::CreateReservedWordParser(_T("if")) 
                            << aslf::AxisGrammar::CreateReservedWordParser(_T("not"))
                            << aslf::AxisGrammar::CreateReservedWordParser(_T("defined")) 
                            << _clause_rule;
	_elseifnot_clause_rule    << aslf::AxisGrammar::CreateReservedWordParser(_T("else")) 
                            << aslf::AxisGrammar::CreateReservedWordParser(_T("if")) 
                            << aslf::AxisGrammar::CreateReservedWordParser(_T("not")) 
                            << aslf::AxisGrammar::CreateReservedWordParser(_T("defined")) 
                            << _clause_rule;
	_conditionals_clause_rule << _if_clause_rule << _ifnot_clause_rule 
                            << _elseif_clause_rule << _elseifnot_clause_rule;

	// add actions
	_clause_rule.AddAction(asla::CallbackAction(*this));
}

axis::String aapp::PreProcessor::ReadLine(void)
{
	String poppedLine;
	String lineToParse;
	if (IsEOF())
	{	// can't read past end of the stream stack
		throw axis::foundation::IOException();
	}
	// parse until we process every file in the input stack
	while (!IsEOF() && lineToParse.empty())
	{
		if (_formatter.IsEOF() || _fullStop || _skipFile)
		{	// we found the end of the current stream go back one level
			// check if all if blocks were closed

			// any emergency break forces file skipping, so correct nesting
			// check is not necessary in these cases
			if (!(_fullStop || _skipFile))
			{
				if (_nestedCondCount != 0 || _ifScanning || _scanEndIfBlock)
				{
					_parseContext.RegisterEvent(asmm::ErrorMessage(0x300601, AXIS_ERROR_MSG_ENDIF_MISSING));
					SetErrorState(ParsingError);
				}
			}
			_inputStack.CloseTopStream();
			if (_inputStack.Count() > 0)	// in case we found an emergency break in the root file...
			{
				_formatter.ChangeSource(_inputStack.GetTopStream());
        // update source information
        _parseContext.SetParseSourceName(_formatter.GetCurrentSource().GetStreamPath());
			}

			// pop back nested block counter
			_nestedCondCount = _nestedCondStack.front();
			_nestedCondStack.pop_front();

			// reset skip flag
			_skipFile = false;
		}
		else
		{
			// pop out line from the formatter
			_formatter.ReadLine(poppedLine);
			lineToParse = ProcessInstruction(poppedLine);
      // update line source information
      _parseContext.SetParseSourceCursorLocation(_formatter.GetLastLineReadIndex());
		}
	}

	if (IsEOF())
	{
		// final check: let's see if everything went ok (if clauses closed correctly)
		if (_nestedCondCount != 0 || _ifScanning || _scanEndIfBlock)
		{
			// error: if without endif
			_parseContext.RegisterEvent(asmm::ErrorMessage(0x300601, AXIS_ERROR_MSG_ENDIF_MISSING));
			SetErrorState(ParsingError);
		}
	}

	return lineToParse;
}

void aapp::PreProcessor::Prepare( void )
{
	// load stream formatter to strip out the comments
	_formatter.ChangeSource(_inputStack.GetTopStream());
  _parseContext.SetParseSourceName(_formatter.GetCurrentSource().GetStreamPath());
}

bool aapp::PreProcessor::IsEOF( void ) const
{
	return (_formatter.IsEOF() && _inputStack.Count() <= 1) || 
          _fullStop || (_skipFile && _inputStack.Count() == 1);
}

bool aapp::PreProcessor::IsSymbolDefined( const axis::String& symbolName ) const
{
	return _symbolTable.IsDefined(symbolName);
}

void aapp::PreProcessor::ProcessLexerSuccessEvent( const aslp::ParseResult& result )
{
	_validationCondition = result.GetParseTree().BuildExpressionString();
}

aapp::PreProcessor::ErrorState aapp::PreProcessor::GetErrorState( void ) const
{
	return _errorState;
}

void aapp::PreProcessor::SetErrorState( ErrorState state )
{
	if ((int)_errorState < (int)state)
	{
		_errorState = state;
	}
}

void aapp::PreProcessor::SetBaseIncludePath( const axis::String& includePath )
{
	_baseIncludePath = includePath;
}

axis::String aapp::PreProcessor::GetAbsolutePath( const axis::String& filename ) const
{
	// we will try to make this an absolute path, if needed and possible
	if (asio::FileSystem::IsAbsolutePath(filename)) return filename;
	if (_baseIncludePath.empty()) return filename;
	String s = asio::FileSystem::ConcatenatePath(_baseIncludePath, filename);
	return asio::FileSystem::ToCanonicalPath(s);
}

void aapp::PreProcessor::AddPreProcessorSymbol( const axis::String& symbolId )
{
	_symbolTable.AddSymbol(*new Symbol(afd::kId, symbolId));
}

void aapp::PreProcessor::ClearPreProcessorSymbols( void )
{
	_symbolTable.ClearTable();
}

unsigned long aapp::PreProcessor::GetLastLineReadIndex( void ) const
{
	return _formatter.GetLastLineReadIndex();
}

axis::String aapp::PreProcessor::GetLastLineSourceName( void ) const
{
	return _formatter.GetCurrentSource().GetStreamPath();
}

void aapp::PreProcessor::Reset( void )
{
	_nestedCondCount = 0;
	_nestedSkippedCondCount = 0;
	_nestedCondStack.clear();
	_ifScanning = false;
	_scanEndIfBlock = false;
	_validationCondition.clear();
	_skipFile = false;
	_fullStop = false;
	_errorState = NoError;
	_formatter.Reset();
}

axis::String aapp::PreProcessor::ProcessInstruction( const axis::String& input )
{
	String outputBuffer;

	// check for a preprocessor command
  asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(input);
  asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(input.end());
	aslp::ParseResult result  = _preprocessor_rule(begin, end);

	if (result.IsMatch())
	{	// found a preprocessor directive. Pass to each handler until someone
		// process the instruction
		String directive = result.GetLastReadPosition().ToString(end);
		if (HandlePreProcessorDirective(outputBuffer, directive))
		{	// the expansion forwarded data to be handled by the parser
			return outputBuffer;
		}
	}
	else
	{	// probably some sort of instruction; let the parser handle it
		if(_ifScanning || _scanEndIfBlock)
		{	// this instruction belongs to a block which we shouldn't process
			return _T("");
		}

		return input;
	}

	// no forwarded data was thrown
	return _T("");
}

bool aapp::PreProcessor::HandlePreProcessorDirective(axis::String& informationOutput, 
                                                     const axis::String& input)
{
	// is it an if clause?
	if(TestAndExecuteIfClause(input))
	{	// if block processed.
		return false;
	}

	if(_ifScanning || _scanEndIfBlock)
	{	// it is not an if clause, but this instruction belongs to a block
		// which we shouldn't process
		return false;
	}

	// it is not an if clause, let's test for each other directive type until
	// we find a match
	if (TestAndExecuteIncludeDirective(input)) return false;
	if (TestAndExecuteDefineDirective(input)) return false;
	if (TestAndExecuteBreakDirectives(input)) return false;

	// until now, this function never returns true because there is no directive
	// that requires parser post-processing; however, if it might be required
	// in the future, the processor can be told to do so by making this function
	// return true and writing output data to the informationOutput parameter.

	// no matches; invalid directive
	_parseContext.RegisterEvent(asmm::ErrorMessage(0x300602, AXIS_ERROR_MSG_UNKNOWN_DIRECTIVE));
	SetErrorState(ParsingError);
	return false;
}

bool aapp::PreProcessor::TestAndExecuteIncludeDirective( const axis::String& input )
{
	String streamDescriptor;

	asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(input);
	asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(input.end());
	aslp::ParseResult result  = _include_rule(begin, end);

	if (result.IsMatch() && result.GetLastReadPosition() == end)
	{	// we've got a full match; it really is an include directive
		if (!_inputStack.CanStore())
		{	// wait, the stack is already full...
			_parseContext.RegisterEvent(asmm::ErrorMessage(0x300603, AXIS_ERROR_MSG_INPUT_STACK_OVERFLOW));
			SetErrorState(CriticalError);

			// this is a severe processing error, abort operation
			_fullStop = true;

			return true;
		}

		// get the filename
		aslp::ExpressionNode& root = (aslp::ExpressionNode&)result.GetParseTree();
		const aslp::ParseTreeNode& fileNameNode = *root.GetLastChild();
		streamDescriptor = fileNameNode.ToString();
		String fullPath = GetAbsolutePath(streamDescriptor);

		try
		{
			// try to open the new file
			_inputStack.AddStream(fullPath);

			// file opened; push nested conditional count to stack
			_nestedCondStack.push_front(_nestedCondCount);
			_nestedCondCount = 0;

			// rewind formatter stack and update context
			_formatter.Rewind();
			_formatter.ChangeSource(_inputStack.GetTopStream());
      // update source information
      _parseContext.SetParseSourceName(_formatter.GetCurrentSource().GetStreamPath());
		}
		catch (axis::foundation::AxisException&)
		{	// something went wrong opening the stream
			// tell to stop processing current file
			_skipFile = true;

			// notify error
			_parseContext.RegisterEvent(
        asmm::ErrorMessage(0x300604, AXIS_ERROR_MSG_INCLUDE_FILE_NOT_FOUND + fullPath));
			SetErrorState(IncludeFileSkipped);
		}
		return true;
	}
	else if (result.GetLastReadPosition() != begin)
	{	// that is, indeed it is our directive, but there are syntax errors
		_parseContext.RegisterEvent(asmm::ErrorMessage(0x300605, AXIS_ERROR_MSG_DIRECTIVE_SYNTAX_ERROR));
		SetErrorState(ParsingError);

		// we have processed it, even though with errors
		return true;
	}
	return false;
}

bool aapp::PreProcessor::TestAndExecuteDefineDirective( const axis::String& input )
{
	String symbolName;
	asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(input);
	asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(input.end());
	aslp::ParseResult result  = _define_rule(begin, end);

	if (result.IsMatch() && result.GetLastReadPosition() == end)
	{	// we've got a full match; it really is a define directive
		// get the enumeration node containing all defined symbols
		aslp::ExpressionNode& root = (aslp::ExpressionNode&)result.GetParseTree();
		const aslp::EnumerationExpression& symbolEnum = 
      (const aslp::EnumerationExpression&)(*root.GetLastChild());
		const aslp::ParseTreeNode *symbolNode = symbolEnum.GetFirstChild();

		// iterate through all definitions
		while(symbolNode != NULL)
		{
			symbolName = symbolNode->ToString();
			if (_symbolTable.IsDefined(symbolName))
			{	// oh oh, the symbol already exists
				_parseContext.RegisterEvent(
          asmm::ErrorMessage(0x300606, AXIS_ERROR_MSG_REDEFINED_DIRECTIVE_FLAG + symbolName));
				SetErrorState(ParsingError);
			}
			else
			{
				_symbolTable.AddSymbol(*new Symbol(afd::kId, symbolName));
			}
			symbolNode = symbolNode->GetNextSibling();
		}
		return true;
	}
	else if (result.GetLastReadPosition() != begin)
	{	// that is, indeed it is our directive, but there are syntax errors
		_parseContext.RegisterEvent(asmm::ErrorMessage(0x300605, AXIS_ERROR_MSG_DIRECTIVE_SYNTAX_ERROR));
		SetErrorState(ParsingError);
		// we have processed it, even though with errors
		return true;
	}
	return false;
}

void aapp::PreProcessor::ValidateExistenceClause( const axis::String& input )
{
	// check if might be any if clause
	asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(input);
	asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(input.end());
	aslp::ParseResult result  = _conditionals_clause_rule(begin, end);

	if (result.IsMatch() && result.GetLastReadPosition() == end)
	{	// it is an if clause and we successfully extract the clause, check it
		String clause = _validationCondition;
		if (!_existenceParser->IsSyntacticallyValid(clause))
		{	// something wrong with the syntax
			throw (axis::foundation::InvalidSyntaxException() << 
        axis::foundation::StandardTraceHints::PreProcessorControl);
		}
	}
}

bool aapp::PreProcessor::TestAndExecuteIfClause( const axis::String& input )
{
	bool ifClause, elseIfClause, endIfSentence, elseClause;
	bool aux1, aux2;
	bool possibleErrorDetected = false;

	String ifExpression, elseIfExpression;
  asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(input);
  asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(input.end());
	bool expectedResult = true;

	// check for each type of clause
	aslp::ParseResult result = _if_clause_rule(begin, end);
	aux1 = result.IsMatch() && result.GetLastReadPosition() == end;
	possibleErrorDetected = possibleErrorDetected || (!result.IsMatch() && result.GetLastReadPosition() != begin);
	if (aux1) ifExpression = _validationCondition;

	result = _ifnot_clause_rule(begin, end);
	aux2 = result.IsMatch() && result.GetLastReadPosition() == end;
	possibleErrorDetected = possibleErrorDetected || (!result.IsMatch() && result.GetLastReadPosition() != begin);
	if (aux2) ifExpression = _validationCondition;

	ifClause = aux1 || aux2;
	expectedResult = aux1? true : false;

	result = _elseif_clause_rule(begin, end);
	aux1 = result.IsMatch() && result.GetLastReadPosition() == end;
	possibleErrorDetected = possibleErrorDetected || (!result.IsMatch() && result.GetLastReadPosition() != begin);
	if (aux1) elseIfExpression = _validationCondition;

	result = _elseifnot_clause_rule(begin, end);
	aux2 = result.IsMatch() && result.GetLastReadPosition() == end;
	possibleErrorDetected = possibleErrorDetected || (!result.IsMatch() && result.GetLastReadPosition() != begin);
	if (aux1) elseIfExpression = _validationCondition;

	elseIfClause = aux1 || aux2;
	expectedResult = (elseIfClause)? (aux1? true : false) : expectedResult;

	result = _endif_clause_rule(begin, end);
	endIfSentence = result.IsMatch() && result.GetLastReadPosition() == end;

	result = _else_clause_rule(begin, end);
	elseClause = result.IsMatch() && result.GetLastReadPosition() == end;

	if (!(ifClause || elseIfClause || endIfSentence || elseClause) && possibleErrorDetected)
	{	// none of the clauses definitions could be completely matched; syntax error detected
		_parseContext.RegisterEvent(asmm::ErrorMessage(0x300605, AXIS_ERROR_MSG_DIRECTIVE_SYNTAX_ERROR));
		SetErrorState(ParsingError);
		return true;
	}

	if (_ifScanning || _scanEndIfBlock)
	{
		if (ifClause)
		{
			_nestedSkippedCondCount++;
			return true;
		}
		if (endIfSentence && _nestedSkippedCondCount > 0)
		{
			_nestedSkippedCondCount--;
			return true;
		}
	}

	if(_nestedSkippedCondCount != 0) return true;

	if (endIfSentence)
	{
		// if we have found the end of a block, we have two alternatives...
		if (_ifScanning)
		{	// 1) no matching expressions found, reactivate normal processing
			_ifScanning = false;
		}
		else
		{	// 2) we were not scanning, so just close one nested block
			_nestedCondCount--;
			if (_nestedCondCount < 0)
			{	// this shouldn't have happened
				_parseContext.RegisterEvent(asmm::ErrorMessage(0x300607, AXIS_ERROR_MSG_ENDIF_IN_EXCESS));
				SetErrorState(ParsingError);
				return true;
			}
		}
		_scanEndIfBlock = false;
		return true;
	}

	// independently of which clause is, check if it's syntactically correct
	if ((ifClause || elseIfClause) && !_ifScanning)
	{
		ValidateExistenceClause(input);
	}

	// we didn't find the "end if" clause yet and the actual block was already
	// processed, so just ignore all the lines until we find the "end if"
	if (_scanEndIfBlock)
	{
		return true;
	}

	if (ifClause && !_ifScanning)
	{
		// check if the expression evaluates to true
		if (EvaluateExistenceClause(ifExpression, expectedResult))
		{	// validation ok, open a new nested clause
			_nestedCondCount++;
			_nestedSkippedCondCount = 0;
			return true;
		}
		else
		{	// not ok, search for another clause or the end of the block
			_ifScanning = true;
			return true;
		}
	}
	else if (elseIfClause)
	{	// we found an alternative existence clause, let's check it
		// do we have to jump or Evaluate it?
		if (_ifScanning)
		{	// Evaluate it
			if(EvaluateExistenceClause(elseIfExpression, expectedResult))
			{	// enter else if block here
				_nestedCondCount++;
				_nestedSkippedCondCount = 0;
				_ifScanning = false;	// exit scanning block mode
			}
			return true;
		}
		else
		{	// no, just go directly to the end if clause
			_scanEndIfBlock = true;
			return true;
		}
	}
	else if(elseClause)
	{	// we found an alternative existence clause, do we have to jump or Evaluate it?
		if (_ifScanning)
		{	// enter else block here
			_nestedCondCount++;
			_nestedSkippedCondCount = 0;
			_ifScanning = false;	// exit scanning block mode
			return true;
		}
		else if(_nestedCondCount > 0)
		{	// no, just go directly to the end if clause
			_scanEndIfBlock = true;
			return true;
		}
		else
		{
			// it seems that we have a dangling else; notify and ignore directive
			_parseContext.RegisterEvent(asmm::ErrorMessage(0x300608, AXIS_ERROR_MSG_ELSE_WITHOUT_IF));
			SetErrorState(ParsingError);
			return true;
		}
	}

	return _ifScanning;
}

bool aapp::PreProcessor::EvaluateExistenceClause( const axis::String& input, bool expectedResult )
{
	if(_existenceParser->Evaluate(input, expectedResult))
	{
		return _existenceParser->GetLastResult();
	}
	else
	{	// we couldn't evaluate, abort
		throw (InvalidSyntaxException() << axis::foundation::StandardTraceHints::PreProcessorControl);
	}
}

bool aapp::PreProcessor::TestAndExecuteBreakDirectives( const axis::String& input )
{
  asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(input);
  asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(input.end());
	bool isSkipDirective, isEndDirective;

	aslp::ParseResult result = _skip_rule(begin, end);
	isSkipDirective = result.IsMatch() && result.GetLastReadPosition() == end;

	result = _end_rule(begin, end);
	isEndDirective = result.IsMatch() && result.GetLastReadPosition() == end;
	_skipFile = isSkipDirective;
	_fullStop = _fullStop || isEndDirective;

	return isSkipDirective || isEndDirective;
}
