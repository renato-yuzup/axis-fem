#if defined DEBUG || defined _DEBUG

#include "unit_tests/unit_tests.hpp"
#include "System.hpp"
#include "AxisString.hpp"
#include "application/parsing/core/StatementDecoder.hpp"
#include "application/parsing/core/ParseContextConcrete.hpp"
#include "application/parsing/core/EventStatistic.hpp"
#include "MockBlockParser.hpp"
#include "application/parsing/error_messages.hpp"


#define TEST_FILE_INPUTPARSER_PATH		"input_test.axis"

using namespace axis::application::parsing::core;
using namespace axis::foundation;
using namespace axis;

// These specializations are required for the use of equality asserts in 
// the MS Unit Test Framework.
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const StatementDecoder::ErrorState& v)
{
	return String::int_parse((long)v).data();
}

namespace axis { namespace unit_tests { namespace core {

	/*
		Checks if the input parser can correctly interpret the basic structure of an input file.
	*/
	TEST_CLASS(StatementDecoderTest)
	{
	public:
    TEST_METHOD_INITIALIZE(SetUp)
    {
      axis::System::Initialize();
    }

    TEST_METHOD_CLEANUP(TearDown)
    {
      axis::System::Finalize();
    }

		TEST_METHOD(TestSimpleRead)
		{
			ParseContextConcrete context;
			StatementDecoder bip(context);
			MockBlockParser mockParser;

			// init parser context
			bip.SetCurrentContext(mockParser);

      bip.FeedDecoderBuffer(_T("LINE 1"));
			bip.ProcessLine();

			Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(0, mockParser.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, bip.IsBufferEmpty());
			Assert::AreEqual(true, bip.GetBufferContents().empty());
			Assert::AreEqual(StatementDecoder::NoError, bip.GetErrorState());
		}

		TEST_METHOD(TestNestedBlocksRead)
		{
			ParseContextConcrete context;
			StatementDecoder bip(context);
			MockBlockParser mockParser1;
			MockBlockParser &mockParser2 = *new MockBlockParser();

			// init parser context
			mockParser1.SetExpectedNestedBlock(_T("TEST"), mockParser2);
			bip.SetCurrentContext(mockParser1);

      bip.FeedDecoderBuffer(_T("LINE 1"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("BEGIN TEST"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("LINE 2"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(0, mockParser2.GetLastLineRead().compare(_T("LINE 2")));

      bip.FeedDecoderBuffer(_T("END TEST"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			// at this time, mockParser2 has already been destroyed

      bip.FeedDecoderBuffer(_T("LINE 3"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 3")));

			Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(true, bip.IsBufferEmpty());
			Assert::AreEqual(true, bip.GetBufferContents().empty());
			Assert::AreEqual(StatementDecoder::NoError, bip.GetErrorState());
		}

		TEST_METHOD(TestParameterizedBlockRead)
		{
			ParseContextConcrete context;
			StatementDecoder bip(context);
			MockBlockParser mockParser1;
			MockBlockParser &mockParser2 = *new MockBlockParser();

			// init parser context
			mockParser1.SetExpectedNestedBlock(_T("TEST"), mockParser2);
			bip.SetCurrentContext(mockParser1);

      bip.FeedDecoderBuffer(_T("LINE 1"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("BEGIN TEST WITH ID=1,MYPROP=HELLO"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			const axis::services::language::syntax::evaluation::ParameterList& paramList = mockParser1.GetParamListReceived();
			Assert::AreEqual(true, paramList.IsDeclared(_T("ID")));
			Assert::AreEqual(true, paramList.IsDeclared(_T("MYPROP")));
			Assert::AreEqual(0, paramList.GetParameterValue(_T("ID")).ToString().compare(_T("1")));
			Assert::AreEqual(0, paramList.GetParameterValue(_T("MYPROP")).ToString().compare(_T("HELLO")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("LINE 2"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(0, mockParser2.GetLastLineRead().compare(_T("LINE 2")));

      bip.FeedDecoderBuffer(_T("END TEST"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			// at this time, mockParser2 has already been destroyed

      bip.FeedDecoderBuffer(_T("LINE 3"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 3")));

			Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(true, bip.IsBufferEmpty());
			Assert::AreEqual(true, bip.GetBufferContents().empty());
			Assert::AreEqual(StatementDecoder::NoError, bip.GetErrorState());
		}

		TEST_METHOD(TestWrappedBlockHeadRead)
		{
			ParseContextConcrete context;
			StatementDecoder bip(context);
			MockBlockParser mockParser1;
			MockBlockParser &mockParser2 = *new MockBlockParser();

			// init parser context
			mockParser1.SetExpectedNestedBlock(_T("TEST"), mockParser2);
			bip.SetCurrentContext(mockParser1);

      bip.FeedDecoderBuffer(_T("LINE 1"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("BEGIN TEST WITH "));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());
			Assert::AreEqual(true, !bip.IsBufferEmpty());
			Assert::AreEqual(0, bip.GetBufferContents().compare(_T("BEGIN TEST WITH ")));

      bip.FeedDecoderBuffer(_T("ID=1, MYPROP="));
			bip.ProcessLine();
			Assert::AreEqual(0, bip.GetBufferContents().compare(_T("BEGIN TEST WITH  ID=1, MYPROP=")));

      bip.FeedDecoderBuffer(_T("HELLO"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			const axis::services::language::syntax::evaluation::ParameterList& paramList = mockParser1.GetParamListReceived();
			Assert::AreEqual(true, paramList.IsDeclared(_T("ID")));
			Assert::AreEqual(true, paramList.IsDeclared(_T("MYPROP")));
			Assert::AreEqual(0, paramList.GetParameterValue(_T("ID")).ToString().compare(_T("1")));
			Assert::AreEqual(0, paramList.GetParameterValue(_T("MYPROP")).ToString().compare(_T("HELLO")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("LINE 2"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(0, mockParser2.GetLastLineRead().compare(_T("LINE 2")));

      bip.FeedDecoderBuffer(_T("END TEST"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			// at this time, mockParser2 has already been destroyed

      bip.FeedDecoderBuffer(_T("LINE 3"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 3")));

			Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(true, bip.IsBufferEmpty());
			Assert::AreEqual(true, bip.GetBufferContents().empty());
			Assert::AreEqual(StatementDecoder::NoError, bip.GetErrorState());
		}

		TEST_METHOD(TestWrappedBlockTailRead)
		{
			ParseContextConcrete context;
			StatementDecoder bip(context);
			MockBlockParser mockParser1;
			MockBlockParser &mockParser2 = *new MockBlockParser();

			// init parser context
			mockParser1.SetExpectedNestedBlock(_T("TEST"), mockParser2);
			bip.SetCurrentContext(mockParser1);

      bip.FeedDecoderBuffer(_T("LINE 1"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("BEGIN TEST"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("LINE 2"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(0, mockParser2.GetLastLineRead().compare(_T("LINE 2")));

      bip.FeedDecoderBuffer(_T("END"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(0, mockParser2.GetLastLineRead().compare(_T("LINE 2")));
			Assert::AreEqual(true, !bip.IsBufferEmpty());

      bip.FeedDecoderBuffer(_T("TEST"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, bip.IsBufferEmpty());
			// at this time, mockParser2 has already been destroyed

      bip.FeedDecoderBuffer(_T("LINE 3"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 3")));

			Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(true, bip.IsBufferEmpty());
			Assert::AreEqual(true, bip.GetBufferContents().empty());
			Assert::AreEqual(StatementDecoder::NoError, bip.GetErrorState());
		}

		TEST_METHOD(TestErrorHeaderSyntax)
		{
			ParseContextConcrete context;
			StatementDecoder bip(context);
			MockBlockParser mockParser1;
			MockBlockParser &mockParser2 = *new MockBlockParser();

			// init parser context
			mockParser1.SetExpectedNestedBlock(_T("TEST"), mockParser2);
			bip.SetCurrentContext(mockParser1);

      bip.FeedDecoderBuffer(_T("LINE 1"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("BEGIN 01"));
			bip.ProcessLine();	// cannot begin block name with numbers
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());
			Assert::AreEqual(StatementDecoder::ParseError, bip.GetErrorState());
			Assert::AreEqual(true, bip.IsBufferEmpty());	// the invalid token must have been disposed
			Assert::AreEqual(true, context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(AXIS_ERROR_ID_MISPLACED_SYMBOL, (int)context.EventSummary().GetLastEventId());
			Logger::WriteMessage(_T("The following error was registered by the input parser:"));
			Logger::WriteMessage(context.EventSummary().GetLastEvent().GetDescription());

      bip.FeedDecoderBuffer(_T("BEGIN %"));
			bip.ProcessLine();	// invalid char sequence
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());
			Assert::AreEqual(StatementDecoder::ParseError, bip.GetErrorState());
			Assert::AreEqual(true, bip.IsBufferEmpty());	// the invalid token must have been disposed
			Assert::AreEqual(true, context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(AXIS_ERROR_ID_INVALID_CHAR_SEQUENCE, (int)context.EventSummary().GetLastEventId());
			Logger::WriteMessage(_T("The following error was registered by the input parser:"));
			Logger::WriteMessage(context.EventSummary().GetLastEvent().GetDescription());

      bip.FeedDecoderBuffer(_T("LINE 2"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 2")));
			Assert::AreEqual(0, mockParser2.GetLastLineRead().compare(_T("")));
		}

		TEST_METHOD(TestErrorTailSyntax)
		{
			ParseContextConcrete context;
			StatementDecoder bip(context);
			MockBlockParser mockParser1;
			MockBlockParser &mockParser2 = *new MockBlockParser();

			// init parser context
			mockParser1.SetExpectedNestedBlock(_T("TEST"), mockParser2);
			bip.SetCurrentContext(mockParser1);

      bip.FeedDecoderBuffer(_T("LINE 1"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("BEGIN TEST"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("LINE 2"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(0, mockParser2.GetLastLineRead().compare(_T("LINE 2")));

      bip.FeedDecoderBuffer(_T("END 01"));
			bip.ProcessLine();	// invalid char sequence
			Assert::AreEqual(StatementDecoder::ParseError, bip.GetErrorState());
			Assert::AreEqual(true, bip.IsBufferEmpty());	// the invalid token must have been disposed
			Assert::AreEqual(true, context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(AXIS_ERROR_ID_MISPLACED_SYMBOL, (int)context.EventSummary().GetLastEventId());
			Logger::WriteMessage(_T("The following error was registered by the input parser:"));
			Logger::WriteMessage(context.EventSummary().GetLastEvent().GetDescription());

      bip.FeedDecoderBuffer(_T("LINE 3"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 3")));
		}

		TEST_METHOD(TestErrorIncompleteBlock)
		{
			ParseContextConcrete context;
			StatementDecoder bip(context);
			MockBlockParser &mockParser1 = *new MockBlockParser();;
			MockBlockParser &mockParser2 = *new MockBlockParser();

			// init parser context
			mockParser1.SetExpectedNestedBlock(_T("TEST"), mockParser2);
			bip.SetCurrentContext(mockParser1);

      bip.FeedDecoderBuffer(_T("LINE 1"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("END BOGUS"));
			bip.ProcessLine();	// END without BEGIN
			Assert::AreEqual(StatementDecoder::ParseError, bip.GetErrorState());
			Assert::AreEqual(true, bip.IsBufferEmpty());	// the invalid statement must have been disposed
			Assert::AreEqual(true, context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(AXIS_ERROR_ID_TAIL_IN_EXCESS, (int)context.EventSummary().GetLastEventId());
			Logger::WriteMessage(_T("The following error was registered by the input parser:"));
			Logger::WriteMessage(context.EventSummary().GetLastEvent().GetDescription());

      bip.FeedDecoderBuffer(_T("BEGIN TEST"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("LINE 2"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(0, mockParser2.GetLastLineRead().compare(_T("LINE 2")));

			// end processing without closing 'TEST' block
			bip.EndProcessing();

			Assert::AreEqual(StatementDecoder::ParseError, bip.GetErrorState());
			Assert::AreEqual(true, bip.IsBufferEmpty());	// the invalid token must have been disposed
			Assert::AreEqual(true, context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(AXIS_ERROR_ID_TAIL_MISSING, (int)context.EventSummary().GetLastEventId());
			Logger::WriteMessage(_T("The following error was registered by the input parser:"));
			Logger::WriteMessage(context.EventSummary().GetLastEvent().GetDescription());
		}

		TEST_METHOD(TestErrorDelimiterMismatch)
		{
			ParseContextConcrete context;
			StatementDecoder bip(context);
			MockBlockParser &mockParser1 = *new MockBlockParser();
			MockBlockParser &mockParser2 = *new MockBlockParser();

			// init parser context
			mockParser1.SetExpectedNestedBlock(_T("TEST"), mockParser2);
			bip.SetCurrentContext(mockParser1);

      bip.FeedDecoderBuffer(_T("LINE 1"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("BEGIN TEST"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("LINE 2"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(0, mockParser2.GetLastLineRead().compare(_T("LINE 2")));

      bip.FeedDecoderBuffer(_T("END BOGUS"));
			bip.ProcessLine();	// delimiter mismatch
			Assert::AreEqual(StatementDecoder::ParseError, bip.GetErrorState());
			Assert::AreEqual(true, bip.IsBufferEmpty());	// the invalid statement must have been disposed
			Assert::AreEqual(true, context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(AXIS_ERROR_ID_BLOCK_DELIMITER_MISMATCH, (int)context.EventSummary().GetLastEventId());
			Logger::WriteMessage(_T("The following error was registered by the input parser:"));
			Logger::WriteMessage(context.EventSummary().GetLastEvent().GetDescription());

      bip.FeedDecoderBuffer(_T("LINE 3"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 3")));
		}

		TEST_METHOD(TestErrorBufferOverflow)
		{
			ParseContextConcrete context;
			StatementDecoder bip(context);
			MockBlockParser mockParser1;

			// init parser context
			bip.SetCurrentContext(mockParser1);

			// set input parser to a ridiculous max buffer length to enforce error
			bip.SetMaxBufferLength(1);

      bip.FeedDecoderBuffer(_T("LINE 1"));
			bip.ProcessLine();
			Assert::AreEqual(true, mockParser1.GetLastLineRead().empty());

			Assert::AreEqual(StatementDecoder::CriticalError, bip.GetErrorState());
			Assert::AreEqual(true, bip.IsBufferEmpty());	// the invalid statement must have been disposed
			Assert::AreEqual(true, context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(AXIS_ERROR_ID_STATEMENT_TOO_LONG, (int)context.EventSummary().GetLastEventId());
			Logger::WriteMessage(_T("The following error was registered by the input parser:"));
			Logger::WriteMessage(context.EventSummary().GetLastEvent().GetDescription());
		}

		TEST_METHOD(TestErrorUnexpectedParserBehavior1)
		{
			ParseContextConcrete context;
			StatementDecoder bip(context);
			MockBlockParser mockParser1;
			MockBlockParser &mockParser2 = *new MockBlockParser(true, true, false, false);

			// init parser context
			mockParser1.SetExpectedNestedBlock(_T("TEST"), mockParser2);
			bip.SetCurrentContext(mockParser1);

      bip.FeedDecoderBuffer(_T("LINE 1"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("BEGIN TEST"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());
			Assert::AreEqual(StatementDecoder::CriticalError, bip.GetErrorState());
			Assert::AreEqual(true, bip.IsBufferEmpty());	// the invalid statement must have been disposed
			Assert::AreEqual(true, context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(AXIS_ERROR_ID_UNDEFINED_OPEN_PARSER_BEHAVIOR, (int)context.EventSummary().GetLastEventId());
			Logger::WriteMessage(_T("The following error was registered by the input parser:"));
			Logger::WriteMessage(context.EventSummary().GetLastEvent().GetDescription());

      bip.FeedDecoderBuffer(_T("LINE 2"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(0, mockParser2.GetLastLineRead().compare(_T("LINE 2")));
		}

		TEST_METHOD(TestErrorUnexpectedParserBehavior2)
		{
			ParseContextConcrete context;
			StatementDecoder bip(context);
			MockBlockParser mockParser1;
			MockBlockParser &mockParser2 = *new MockBlockParser(true, false, true, false);

			// init parser context
			mockParser1.SetExpectedNestedBlock(_T("TEST"), mockParser2);
			bip.SetCurrentContext(mockParser1);

      bip.FeedDecoderBuffer(_T("LINE 1"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("BEGIN TEST"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(StatementDecoder::NoError, bip.GetErrorState());
			Assert::AreEqual(true, bip.IsBufferEmpty());
			Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());

      bip.FeedDecoderBuffer(_T("LINE 2"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(0, mockParser2.GetLastLineRead().compare(_T("LINE 2")));

      bip.FeedDecoderBuffer(_T("END TEST"));
			bip.ProcessLine();
			Assert::AreEqual(StatementDecoder::CriticalError, bip.GetErrorState());
			Assert::AreEqual(true, bip.IsBufferEmpty());	// the invalid statement must have been disposed
			Assert::AreEqual(true, context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(AXIS_ERROR_ID_UNDEFINED_CLOSE_PARSER_BEHAVIOR, (int)context.EventSummary().GetLastEventId());
			Logger::WriteMessage(_T("The following error was registered by the input parser:"));
			Logger::WriteMessage(context.EventSummary().GetLastEvent().GetDescription());

      bip.FeedDecoderBuffer(_T("LINE 3"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 3")));
		}

		TEST_METHOD(TestErrorUnexpectedParserBehavior3)
		{
			ParseContextConcrete context;
			StatementDecoder bip(context);
			MockBlockParser mockParser1;
			MockBlockParser &mockParser2 = *new MockBlockParser(true, false, false, true);

			// init parser context
			mockParser1.SetExpectedNestedBlock(_T("TEST"), mockParser2);
			bip.SetCurrentContext(mockParser1);

      bip.FeedDecoderBuffer(_T("LINE 1"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(true, mockParser2.GetLastLineRead().empty());

      bip.FeedDecoderBuffer(_T("BEGIN TEST"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(StatementDecoder::NoError, bip.GetErrorState());
			Assert::AreEqual(true, bip.IsBufferEmpty());
			Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());

      bip.FeedDecoderBuffer(_T("LINE 2"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(StatementDecoder::CriticalError, bip.GetErrorState());
			Assert::AreEqual(true, bip.IsBufferEmpty());	// the invalid statement must have been disposed
			Assert::AreEqual(true, context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(AXIS_ERROR_ID_UNEXPECTED_PARSER_READ_BEHAVIOR, (int)context.EventSummary().GetLastEventId());
			Logger::WriteMessage(_T("The following error was registered by the input parser:"));
			Logger::WriteMessage(context.EventSummary().GetLastEvent().GetDescription());

      bip.FeedDecoderBuffer(_T("END TEST"));
			bip.ProcessLine();

      bip.FeedDecoderBuffer(_T("LINE 3"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 3")));
		}

		TEST_METHOD(TestErrorUnrecognizedBlock)
		{
			ParseContextConcrete context;
			StatementDecoder bip(context);
			MockBlockParser mockParser1;

			// init parser context
			bip.SetCurrentContext(mockParser1);

      bip.FeedDecoderBuffer(_T("LINE 1"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));

      bip.FeedDecoderBuffer(_T("BEGIN TEST"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 1")));
			Assert::AreEqual(StatementDecoder::ParseError, bip.GetErrorState());
			Assert::AreEqual(true, bip.IsBufferEmpty());	// the invalid statement must have been disposed
			Assert::AreEqual(true, context.EventSummary().HasAnyEventRegistered());
			Assert::AreEqual(AXIS_ERROR_ID_UNKNOWN_BLOCK, (int)context.EventSummary().GetLastEventId());
			Logger::WriteMessage(_T("The following error was registered by the input parser:"));
			Logger::WriteMessage(context.EventSummary().GetLastEvent().GetDescription());

      bip.FeedDecoderBuffer(_T("LINE 2"));
			bip.ProcessLine();
			Assert::AreEqual(0, mockParser1.GetLastLineRead().compare(_T("LINE 2")));
		}
	};


} } }

#endif
