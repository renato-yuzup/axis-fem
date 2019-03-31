#pragma once

#include "InputStack.hpp"
#include "InstructionFeeder.hpp"

#include "application/parsing/preprocessing/SymbolTable.hpp"
#include "application/parsing/preprocessing/ExistenceExpressionParser.hpp"
#include "application/parsing/core/ParseContextConcrete.hpp"
#include "services/language/primitives/GeneralExpressionParser.hpp"
#include "services/language/primitives/OrExpressionParser.hpp"
#include "services/language/primitives/EnumerationParser.hpp"
#include "services/language/actions/ParserCallback.hpp"

namespace axis { namespace application { namespace parsing { namespace preprocessing {

/// <summary>
/// Implements a preprocessor which handles file input and interprets preprocessing directives in the file,
/// passing a transformed output to the parser.
/// </summary>
class PreProcessor : public axis::services::language::actions::ParserCallback
{
public:
	enum ErrorState
	{
		NoError = 0,
		ParsingError = 10,
		IncludeFileSkipped = 20,
		IOError = 40,
		CriticalError = 50,
	};

  /// <summary>
	/// Creates a new instance of this class.
	/// </summary>
	/// <param name="inputStack">Stack of input streams to parse.</param>
	PreProcessor(InputStack& inputStack, axis::application::parsing::core::ParseContextConcrete& context);

	/// <summary>
	/// Destroys this object.
	/// </summary>
	/// <remarks>
	/// When destroyed, this object also destroys its associated 
	/// input stream passed in the constructor call.
	/// </remarks>
	virtual ~PreProcessor(void);

	/**********************************************************************************************//**
		* <summary> Defines a preprocessor symbol.</summary>
		*
		* <param name="symbolId"> Identifier for the symbol.</param>
		**************************************************************************************************/
	void AddPreProcessorSymbol(const axis::String& symbolId);

	/**********************************************************************************************//**
		* <summary> Forgets all preprocessor symbols previously defined.</summary>
		**************************************************************************************************/
	void ClearPreProcessorSymbols(void);
	axis::String ReadLine(void);
	void Prepare(void);
	void Reset(void);
	bool IsEOF(void) const;
	ErrorState GetErrorState(void) const;
	unsigned long GetLastLineReadIndex(void) const;
	axis::String GetLastLineSourceName(void) const;
	bool IsSymbolDefined(const axis::String& symbolName) const;
	virtual void ProcessLexerSuccessEvent(const axis::services::language::parsing::ParseResult& result);
	void SetBaseIncludePath(const axis::String& includePath);
private:
	/// <summary>
	/// Initializes grammar rules.
	/// </summary>
	void InitGrammar(void);

	/// <summary>
	/// Expands a preprocessor directive and execute the corresponding actions associated with it.
	/// </summary>
	/// <param name="informationOutput">Reference to the buffer to which output can be written to.</param>
	/// <param name="input">Directive to be processed.</param>
	/// <returns>
	/// True if there is relevant information left in the output buffer, False otherwise.
	/// </returns>
	virtual bool HandlePreProcessorDirective(axis::String& informationOutput, const axis::String& input);

  /// <summary>
	/// Expands #include directives.
	/// </summary>
	/// <param name="input">Directive to be analysed.</param>
	/// <returns>
	/// True if processing was successful, False otherwise.
	/// </returns>
	bool TestAndExecuteIncludeDirective(const axis::String& input);

	/// <summary>
	/// Expands #define directives.
	/// </summary>
	/// <param name="input">Directive to be analysed.</param>
	/// <returns>
	/// True if processing was successful, False otherwise.
	/// </returns>
	bool TestAndExecuteDefineDirective(const axis::String& input);

	/// <summary>
	/// Verifies if the existence clause of an if directive is correct.
	/// </summary>
	/// <param name="input">Clause to be analysed.</param>
	/// <exception cref="InvalidSyntaxException">If the clause is invalid.</exception>
	void ValidateExistenceClause(const axis::String& input);

	/// <summary>
	/// Evaluates an existence clause.
	/// </summary>
	/// <param name="input">Clause to be evaluated.</param>
	/// <param name="expectedResult">Expected value for the evaluated clause.</param>
	/// <returns>
	/// True if the clause evaluated to the same value as the parameter
	/// expectedResult. False otherwise.
	/// </returns>
	bool EvaluateExistenceClause(const axis::String& input, bool expectedResult);

	/// <summary>
	/// Expands #skip and #end directives.
	/// </summary>
	/// <param name="input">Directive to be analysed.</param>
	/// <returns>
	/// True if processing was successful, False otherwise.
	/// </returns>
	bool TestAndExecuteBreakDirectives( const axis::String& input );

	/// <summary>
	/// Process a preprocessor directive if applicable, post-process
	/// the instruction and forward to the parser handle it.
	/// </summary>
	/// <param name="input">Expression to be evaluated.</param>
	axis::String ProcessInstruction(const axis::String& input);

	/// <summary>
	/// Checks if an expression is an if directive and process
	/// it.
	/// </summary>
	/// <param name="input">Expression to be analysed.</param>
	/// <returns>
	/// True if the clause was an if directive and processing
	/// went successful, False otherwise.
	/// </returns>
	bool TestAndExecuteIfClause( const axis::String& input );

	void SetErrorState(ErrorState state);

  axis::String GetAbsolutePath(const axis::String& filename) const;

  InputStack& _inputStack;		// stack of opened files
  axis::application::parsing::core::ParseContextConcrete& _parseContext;
  axis::application::parsing::preprocessing::SymbolTable _symbolTable;	// symbols defined
  axis::application::parsing::preprocessing::ExistenceExpressionParser *_existenceParser;
  InstructionFeeder _formatter;

  // our grammar rules
  axis::services::language::primitives::GeneralExpressionParser _preprocessor_rule;
  axis::services::language::primitives::GeneralExpressionParser _define_rule;
  axis::services::language::primitives::OrExpressionParser      _alt_define_rule;
  axis::services::language::primitives::EnumerationParser       _symbol_enum_rule;
  axis::services::language::primitives::GeneralExpressionParser _include_rule;
  axis::services::language::primitives::GeneralExpressionParser _skip_rule;
  axis::services::language::primitives::GeneralExpressionParser _end_rule;
  axis::services::language::primitives::GeneralExpressionParser _clause_rule;
  axis::services::language::primitives::OrExpressionParser      _expr_rule;
  axis::services::language::primitives::GeneralExpressionParser _bin_op_expr_rule;
  axis::services::language::primitives::OrExpressionParser      _term_rule;
  axis::services::language::primitives::GeneralExpressionParser _group_rule;
  axis::services::language::primitives::GeneralExpressionParser _un_op_expr_rule;
  axis::services::language::primitives::OrExpressionParser      _bin_op_rule;
  axis::services::language::primitives::OrExpressionParser      _un_op_rule;
  axis::services::language::primitives::OrExpressionParser      _conditionals_clause_rule;
  axis::services::language::primitives::GeneralExpressionParser _if_clause_rule;
  axis::services::language::primitives::GeneralExpressionParser _elseif_clause_rule;
  axis::services::language::primitives::GeneralExpressionParser _else_clause_rule;
  axis::services::language::primitives::GeneralExpressionParser _endif_clause_rule;
  axis::services::language::primitives::GeneralExpressionParser _ifnot_clause_rule;
  axis::services::language::primitives::GeneralExpressionParser _elseifnot_clause_rule;

  // used for nested if defined clauses
  int _nestedCondCount;
  int _nestedSkippedCondCount;
  std::list<int> _nestedCondStack;
  bool _ifScanning;		// indicates that we are searching for a clause that evaluates to true in an if block
  bool _scanEndIfBlock;	// indicates that we have to go directly to the end if clause
  axis::String _validationCondition;	// extracted condition from if clauses

  axis::String _baseIncludePath;

  // flags indicating to skip file processing
  bool _fullStop;
  bool _skipFile;

  // error state flag
  ErrorState _errorState;
};

} } } } // namespace axis::application::parsing::preprocessing
