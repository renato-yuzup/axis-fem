#if defined DEBUG || defined _DEBUG

#include "unit_tests/unit_tests.hpp"

#include "services/io/FileReader.hpp"
#include <sys/stat.h>
#include <direct.h>
#include "application/parsing/preprocessing/PreProcessor.hpp"
#include "AxisString.hpp"
#include "System.hpp"
#include "application/parsing/preprocessing/InputStack.hpp"
#include "application/parsing/error_messages.hpp"
#include "application/parsing/core/ParseContextConcrete.hpp"
#include "application/parsing/core/EventStatistic.hpp"
#include "BogusFileReader.hpp"


using namespace axis::foundation;
using namespace axis::application::parsing::preprocessing;
using namespace axis::application::parsing::core;


#define TEST_FILE_INPUTPARSER_PATH		"input_test.axis"


#define TEST_SIMPLEFILE			"test_preprocessor_simple01.txt"
#define TEST_DEFINEFILE			"test_preprocessor_define01.txt"
#define TEST_IFCLAUSESFILE		"test_preprocessor_ifclauses01.txt"
#define TEST_INCLUDEFILE01		"test_preprocessor_include01.txt"
#define TEST_INCLUDEFILE02		"test_preprocessor_include02.txt"
#define TEST_INCLUDEFILE03		"test_preprocessor_include03.txt"
#define TEST_BREAKFILE01		"test_preprocessor_break01.txt"
#define TEST_BREAKFILE02		"test_preprocessor_break02.txt"
#define TEST_BREAKFILE03		"test_preprocessor_break03.txt"
#define TEST_ERRORUNKNOWN		"test_preprocessor_error_unknown01.txt"
#define TEST_ERRORMANYENDCOMM	"test_preprocessor_error_manyendcomm01.txt"
#define TEST_ERRORSYNTAX		"test_preprocessor_error_syntax01.txt"
#define TEST_ERRORNOFILE		"test_preprocessor_error_filenotfound01.txt"
#define TEST_ERRORNOENDCOMMENT	"test_preprocessor_error_noendcomment01.txt"
#define TEST_ERRORIFDIRECTIVES	"test_preprocessor_error_ifdirectives01.txt"
#define TEST_ERRORIO			"test_preprocessor_error_io01.txt"
#define TEST_ERRORMANYNESTED01	"test_preprocessor_error_manynested01.txt"
#define TEST_ERRORMANYNESTED02	"test_preprocessor_error_manynested02.txt"
#define TEST_ERRORMANYNESTED03	"test_preprocessor_error_manynested03.txt"



namespace axis { namespace unit_tests { namespace core {
  
TEST_CLASS(PreProcessorTest)
{
private:
	static axis::services::io::FileReader *file;
	static axis::application::parsing::preprocessing::PreProcessor *preProcessor;
	static axis::application::parsing::preprocessing::InputStack *inputStack;
	static std::string _testFileBasePath;

	static bool FileExists(std::string filename)
	{
		struct stat fileInfo;

		return (stat(filename.c_str(), &fileInfo) == 0);
	}

	static std::string GetTestFileLocation(const std::string& fileName)
	{
		if (_testFileBasePath.empty())
		{
			char *fullPath = new char[255];
			_getcwd(fullPath, 255);
			_testFileBasePath = fullPath;
			delete fullPath;
		}

		std::string buf = _testFileBasePath;
		if (buf[buf.size() - 1] != '\\' && buf[buf.size() - 1] != '/')
		{
			buf.append("/");	// acceptable on Windows
		}
		buf.append(fileName);
		return buf;
	}

	static std::wofstream& CreateFile(const std::string& fileName)
	{
		if (FileExists(fileName.data()))
		{
			remove(fileName.data());
		}

		std::wofstream *s = new std::wofstream();
		s->open(fileName);
		if (!s->is_open())
		{
			Assert::Fail(_T("Couldn't create test file. SetUp failed!"));
		}

		return *s;
	}

	static void InitInputStack(const std::string& fileName)
	{
		axis::String str = StringEncoding::ASCIIToUnicode(GetTestFileLocation(fileName).data());

		inputStack = new axis::application::parsing::preprocessing::InputStack();
		inputStack->AddStream(str);
	}
public:
	/*
		Creates a test input file and set up test objects.
	*/
	TEST_METHOD_INITIALIZE(SetUp)
	{
    axis::System::Initialize();
		CreateSimpleTestFile();
		CreateDefineTestFile();
		CreateIfClausesTestFile();
		CreateIncludeTestFile();
		CreateBreakTestFile();
		CreateErrorUnknownDirectiveTestFile();
		CreateErrorManyEndCommentTestFile();
		CreateErrorSyntaxTestFile();
		CreateErrorFileNotFoundFile();
		CreateErrorNoEndCommentFile();
		CreateErrorIfErrorsFile();
		CreateErrorIOErrorFile();
		CreateErrorManyNestedFilesFile();
	}

	/*
		Destroys the test input file previously created and test objects.
	*/
	TEST_METHOD_CLEANUP(TearDown)
	{
		remove(GetTestFileLocation(TEST_SIMPLEFILE).data());
		remove(GetTestFileLocation(TEST_DEFINEFILE).data());
		remove(GetTestFileLocation(TEST_IFCLAUSESFILE).data());
		remove(GetTestFileLocation(TEST_INCLUDEFILE01).data());
		remove(GetTestFileLocation(TEST_INCLUDEFILE02).data());
		remove(GetTestFileLocation(TEST_INCLUDEFILE03).data());
		remove(GetTestFileLocation(TEST_BREAKFILE01).data());
		remove(GetTestFileLocation(TEST_BREAKFILE02).data());
		remove(GetTestFileLocation(TEST_BREAKFILE03).data());
		remove(GetTestFileLocation(TEST_ERRORUNKNOWN).data());
		remove(GetTestFileLocation(TEST_ERRORMANYENDCOMM).data());
		remove(GetTestFileLocation(TEST_ERRORSYNTAX).data());
		remove(GetTestFileLocation(TEST_ERRORNOFILE).data());
		remove(GetTestFileLocation(TEST_ERRORNOENDCOMMENT).data());
		remove(GetTestFileLocation(TEST_ERRORIFDIRECTIVES).data());
		remove(GetTestFileLocation(TEST_ERRORIO).data());
		remove(GetTestFileLocation(TEST_ERRORMANYNESTED01).data());
		remove(GetTestFileLocation(TEST_ERRORMANYNESTED02).data());
		remove(GetTestFileLocation(TEST_ERRORMANYNESTED03).data());
    axis::System::Finalize();
	}

	static void CreateSimpleTestFile( void )
	{
		std::wofstream& s = CreateFile(GetTestFileLocation(TEST_SIMPLEFILE));

		s << _T("# THIS IS A COMMENT THAT MUST BE IGNORED BY THE PREPROCESSOR\n")
			 _T("/*\n")
			 _T("\tTHIS IS ALSO A COMMENT THAT\n")
			 _T("\tMUST \n")
			 _T("\tBE IGNORED\n")
			 _T("\tBY THE PREPROCESSOR */\n")
			 _T("\n")
			 _T("THIS IS THE FIRST LINE WHICH SHOULD BE READ\n")
			 _T("ONLY THIS PART MUST BE READ\t# NO PARAMETERS \n")
			 _T("    BOGUS DATA: 1\t12.0\t 14.57 \t 0.0\n")
			 _T("/* COMMENT */ SOME /* COMMENT */ DATA /* COMMENT */ HERE # COMMENT")
			 _T("# THIS IS THE END OF THE FILE");
		s.close();
		delete &s;
	}

	static void CreateDefineTestFile( void )
	{
		std::wofstream& s = CreateFile(GetTestFileLocation(TEST_DEFINEFILE));

		s << _T("# THIS IS A COMMENT THAT MUST BE IGNORED BY THE PREPROCESSOR\n")
			_T("@define __MY_SYMBOL1\n")
			_T("@define MY_SYMBOL2\n")
			_T("BOGUS LINE\n")
			_T("\n")
			_T("/* @define SYMBOL_WHICH_SHOULD_NOT_BE_READ */\n")
			_T("@define MY_SYMBOL3\n");
		s.close();
		delete &s;
	}

	static void CreateIfClausesTestFile( void )
	{
		std::wofstream& s = CreateFile(GetTestFileLocation(TEST_IFCLAUSESFILE));

		s << _T("# THIS IS A COMMENT THAT MUST BE IGNORED BY THE PREPROCESSOR\n")
			_T("@define MY_SYMBOL1\n")
			_T("@define MY_SYMBOL2\n")
			_T("@if defined (MY_SYMBOL1 AND MY_SYMBOL2)\n")
			_T("LINE 1\n")
			_T("LINE 2\n")
			_T("@else\n")
			_T("LINE 3\n")
			_T("@endif\n")
			_T("@if not defined (MY_SYMBOL3)\n")
			_T("LINE 4\n")
			_T("@endif\n")
			_T("@if defined (MY_SYMBOL4)\n")
			_T("LINE 5\n")
			_T("@else if defined (MY_SYMBOL5)\n")
			_T("LINE 6\n")
			_T("@else\n")
			_T("LINE 7\n")
			_T("@endif\n");
		s.close();
		delete &s;
	}

	static void CreateIncludeTestFile( void )
	{
		std::wofstream& s = CreateFile(GetTestFileLocation(TEST_INCLUDEFILE01));
		s << _T("# THIS IS A COMMENT THAT MUST BE IGNORED BY THE PREPROCESSOR\n")
			_T("LINE 1\n")
			_T("@define MY_SYMBOL1\n")
			_T("@include \"test_preprocessor_include02.txt\"\n")
			_T("LINE 5\n")
			_T("LINE 6\n");
		s.close();
		delete &s;

		std::wofstream& s1 = CreateFile(GetTestFileLocation(TEST_INCLUDEFILE02));
		s1 << _T("# THIS IS A COMMENT THAT MUST BE IGNORED BY THE PREPROCESSOR\n")
			_T("@include \"test_preprocessor_include03.txt\"\n")
			_T("LINE 4\n");
		s1.close();
		delete &s1;
		std::wofstream& s2 = CreateFile(GetTestFileLocation(TEST_INCLUDEFILE03));
		s2 << _T("# THIS IS A COMMENT THAT MUST BE IGNORED BY THE PREPROCESSOR\n")
			_T("LINE 2\n")
			_T("LINE 3\n");
		s2.close();
		delete &s2;
	}

	static void CreateBreakTestFile( void )
	{
		std::wofstream& s = CreateFile(GetTestFileLocation(TEST_BREAKFILE01));
		s << _T("# THIS IS A COMMENT THAT MUST BE IGNORED BY THE PREPROCESSOR\n")
			_T("LINE 1\n")
			_T("@include \"test_preprocessor_break02.txt\"\n")
			_T("LINE 8\n")
			_T("LINE 9\n");
		s.close();
		delete &s;

		std::wofstream& s1 = CreateFile(GetTestFileLocation(TEST_BREAKFILE02));
		s1 << _T("# THIS IS A COMMENT THAT MUST BE IGNORED BY THE PREPROCESSOR\n")
			_T("LINE 2\n")
			_T("@include \"test_preprocessor_break03.txt\"\n")
			_T("LINE 5\n")
			_T("LINE 6\n")
			_T("@end\n")
			_T("LINE 7\n");
		s1.close();
		delete &s1;
		std::wofstream& s2 = CreateFile(GetTestFileLocation(TEST_BREAKFILE03));
		s2 << _T("# THIS IS A COMMENT THAT MUST BE IGNORED BY THE PREPROCESSOR\n")
			_T("LINE 3\n")
			_T("@skip\n")
			_T("LINE 4\n");
		s2.close();
		delete &s2;
	}

	static void CreateErrorUnknownDirectiveTestFile( void )
	{
		std::wofstream& s = CreateFile(GetTestFileLocation(TEST_ERRORUNKNOWN));
		s << _T("# THIS IS A COMMENT THAT MUST BE IGNORED BY THE PREPROCESSOR\n")
			_T("LINE 1\n")
			_T("@bogus \"any_bogus_param\"\n")
			_T("@define any_name\n")
			_T("LINE 2\n")
			_T("LINE 3\n");
		s.close();
		delete &s;
	}

	static void CreateErrorManyEndCommentTestFile( void )
	{
		std::wofstream& s = CreateFile(GetTestFileLocation(TEST_ERRORMANYENDCOMM));
		s << _T("# THIS IS A COMMENT THAT MUST BE IGNORED BY THE PREPROCESSOR\n")
			_T("LINE 1\n")
			_T("/* OUR COMMENT STARTS HERE ...\n")
			_T("CONTINUES HERE...\n")
			_T("BUT DOUBLE ENDS HERE... */ */ */ AND MUST READ THIS\n")
			_T("AND READ ALSO HERE\n");
		s.close();
		delete &s;
	}

	static void CreateErrorSyntaxTestFile(void)
	{
		std::wofstream& s = CreateFile(GetTestFileLocation(TEST_ERRORSYNTAX));
		s << _T("# THIS IS A COMMENT THAT MUST BE IGNORED BY THE PREPROCESSOR\n")
			_T("@define\n")
			_T("@define 123456\n")
			_T("LINE 1\n")
			_T("@define my_symbol\n")
			_T("@include abc\n")
			_T("LINE 2\n")
			_T("@if defined ()\n");
		s.close();
		delete &s;
	}

	static void CreateErrorFileNotFoundFile(void)
	{
		std::wofstream& s = CreateFile(GetTestFileLocation(TEST_ERRORNOFILE));
		s << _T("LINE 1\n")
			_T("@include \"we have a bogus file here!.something\"\n")
			_T("LINE 2\n");
		s.close();
		delete &s;
	}

	static void CreateErrorNoEndCommentFile(void)
	{
		std::wofstream& s = CreateFile(GetTestFileLocation(TEST_ERRORNOENDCOMMENT));
		s << _T("/* START COMMENT HERE\n")
			_T("MORE COMMENTS HERE\n")
			_T("AND ENDS HERE WITHOUT PROPER DELIMITER\n");
		s.close();
		delete &s;
	}

	static void CreateErrorIfErrorsFile(void)
	{
		std::wofstream& s = CreateFile(GetTestFileLocation(TEST_ERRORIFDIRECTIVES));
		s << _T("@else\n")
			_T("@define BOGUS_SYMBOL, MY_SYMBOL1, MY_SYMBOL2\n")
			_T("LINE 1\n")
			_T("@if defined (BOGUS_SYMBOL)\n")
			_T("LINE 2\n")
			_T("@endif\n")
			_T("@endif\n")
			_T("LINE 3\n")
			_T("@if defined (BOGUS)\n");
		s.close();
		delete &s;
	}

	static void CreateErrorIOErrorFile(void)
	{
		std::wofstream& s = CreateFile(GetTestFileLocation(TEST_ERRORIO));

		// create just a blank file

		s.close();
		delete &s;
	}

	static void CreateErrorManyNestedFilesFile(void)
	{
		std::wofstream& s1 = CreateFile(GetTestFileLocation(TEST_ERRORMANYNESTED01));
		s1 << _T("LINE 1\n")
			_T("@include \"test_preprocessor_error_manynested02.txt\"\n")
			_T("LINE 2\n");
		s1.close();
		delete &s1;

		std::wofstream& s2 = CreateFile(GetTestFileLocation(TEST_ERRORMANYNESTED02));
		s2 << _T("LINE 3\n")
			_T("@include \"test_preprocessor_error_manynested03.txt\"\n")
			_T("LINE 4");
		s2.close();
		delete &s2;

		std::wofstream& s3 = CreateFile(GetTestFileLocation(TEST_ERRORMANYNESTED03));
		s3 << _T("BOGUS -- MUST NOT READ");
		s3.close();
		delete &s3;
	}

	/*
		Checks if the input parser can correctly interpret the basic structure of an input file.
	*/
	TEST_METHOD(TestSimpleRead)
	{
		String line;

		// init objects
		ParseContextConcrete context;
		InitInputStack(TEST_SIMPLEFILE);
		preProcessor = new axis::application::parsing::preprocessing::PreProcessor(*inputStack, context);

		preProcessor->Prepare();

		Assert::AreEqual(true,!preProcessor->IsEOF());

		// read each line and compare
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("THIS IS THE FIRST LINE WHICH SHOULD BE READ")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("ONLY THIS PART MUST BE READ")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("BOGUS DATA: 1\t12.0\t 14.57 \t 0.0")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("SOME  DATA  HERE")));

		// do final checks
		Assert::AreEqual(true,preProcessor->IsEOF());
		Assert::AreEqual(true,!context.EventSummary().HasAnyEventRegistered());

		delete preProcessor;
		delete inputStack;
	}

	TEST_METHOD(TestDefineDirectives)
	{
		String line;

		// init objects
		ParseContextConcrete context;
		InitInputStack(TEST_DEFINEFILE);
		preProcessor = new axis::application::parsing::preprocessing::PreProcessor(*inputStack, context);

		preProcessor->Prepare();

		Assert::AreEqual(true,!preProcessor->IsEOF());

		// read each line and compare
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("BOGUS LINE")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("")));

		// do final checks
		Assert::AreEqual(true,preProcessor->IsEOF());
		Assert::AreEqual(true,!context.EventSummary().HasAnyEventRegistered());
		Assert::AreEqual(true,preProcessor->IsSymbolDefined(_T("__MY_SYMBOL1")));
		Assert::AreEqual(true,preProcessor->IsSymbolDefined(_T("MY_SYMBOL2")));
		Assert::AreEqual(true,!preProcessor->IsSymbolDefined(_T("SYMBOL_WHICH_SHOULD_NOT_BE_READ")));
		Assert::AreEqual(true,preProcessor->IsSymbolDefined(_T("MY_SYMBOL3")));

		delete preProcessor;
		delete inputStack;
	}

	TEST_METHOD(TestIfClausesDirectives)
	{
		String line;

		// init objects
		ParseContextConcrete context;
		InitInputStack(TEST_IFCLAUSESFILE);
		preProcessor = new axis::application::parsing::preprocessing::PreProcessor(*inputStack, context);

		preProcessor->Prepare();

		Assert::AreEqual(true,!preProcessor->IsEOF());

		// read each line and compare
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 1")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 2")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 4")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 7")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("")));

		// do final checks
		Assert::AreEqual(true,preProcessor->IsEOF());
		Assert::AreEqual(true,!context.EventSummary().HasAnyEventRegistered());
		Assert::AreEqual(true,preProcessor->IsSymbolDefined(_T("MY_SYMBOL1")));
		Assert::AreEqual(true,preProcessor->IsSymbolDefined(_T("MY_SYMBOL2")));
		Assert::AreEqual(true,!preProcessor->IsSymbolDefined(_T("MY_SYMBOL3")));
		Assert::AreEqual(true,!preProcessor->IsSymbolDefined(_T("MY_SYMBOL4")));
		Assert::AreEqual(true,!preProcessor->IsSymbolDefined(_T("MY_SYMBOL5")));

		delete preProcessor;
		delete inputStack;
	}

	TEST_METHOD(TestIncludeDirectives)
	{
		String line;

		// init objects
		ParseContextConcrete context;
		InitInputStack(TEST_INCLUDEFILE01);
		preProcessor = new axis::application::parsing::preprocessing::PreProcessor(*inputStack, context);

		preProcessor->Prepare();

		Assert::AreEqual(true,!preProcessor->IsEOF());

		// read each line and compare
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 1")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 2")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 3")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 4")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 5")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 6")));

		// do final checks
		Assert::AreEqual(true,preProcessor->IsEOF());
		Assert::AreEqual(true,!context.EventSummary().HasAnyEventRegistered());

		delete preProcessor;
		delete inputStack;
	}

	TEST_METHOD(TestEmergencyBreak)
	{
		String line;

		// init objects
		ParseContextConcrete context;
		InitInputStack(TEST_BREAKFILE01);
		preProcessor = new axis::application::parsing::preprocessing::PreProcessor(*inputStack, context);

		preProcessor->Prepare();

		Assert::AreEqual(true,!preProcessor->IsEOF());

		// read each line and compare
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 1")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 2")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 3")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 5")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 6")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("")));

		// do final checks
		Assert::AreEqual(true,preProcessor->IsEOF());
		Assert::AreEqual(true,!context.EventSummary().HasAnyEventRegistered());

		delete preProcessor;
		delete inputStack;
	}

	TEST_METHOD(TestErrorUnknownDirective)
	{
		String line;

		// init objects
		ParseContextConcrete context;
		InitInputStack(TEST_ERRORUNKNOWN);
		preProcessor = new axis::application::parsing::preprocessing::PreProcessor(*inputStack, context);

		preProcessor->Prepare();

		Assert::AreEqual(true,!preProcessor->IsEOF());

		// read each line and compare
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 1")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 2")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 3")));

		// do final checks
		Assert::AreEqual(true,preProcessor->IsEOF());
		Assert::AreEqual(true,preProcessor->IsSymbolDefined(_T("any_name")));
		Assert::AreEqual(true,context.EventSummary().HasAnyEventRegistered());
		Assert::AreEqual(1, (int)context.EventSummary().GetTotalEventCount());
		Assert::AreEqual(AXIS_ERROR_ID_UNKNOWN_DIRECTIVE, (int)context.EventSummary().GetLastEventId());

		delete preProcessor;
		delete inputStack;
	}

	TEST_METHOD(TestErrorManyEndComment)
	{
		String line;

		// init objects
		ParseContextConcrete context;
		InitInputStack(TEST_ERRORMANYENDCOMM);
		preProcessor = new axis::application::parsing::preprocessing::PreProcessor(*inputStack, context);

		preProcessor->Prepare();

		Assert::AreEqual(true,!preProcessor->IsEOF());

		// read each line and compare
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 1")));

		// it must already have noticed that we have mismatched comment delimiters
		// because the stream formatter looks ahead of the position of the
		// preprocessor
		Assert::AreEqual(true,context.EventSummary().HasAnyEventRegistered());

		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("AND MUST READ THIS")));

		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("AND READ ALSO HERE")));

		// do final checks
		Assert::AreEqual(true,preProcessor->IsEOF());
		Assert::AreEqual(true,context.EventSummary().HasAnyEventRegistered());
		Assert::AreEqual(2, (int)context.EventSummary().GetTotalEventCount());
		Assert::AreEqual(AXIS_ERROR_ID_END_COMMENT_IN_EXCESS, (int)context.EventSummary().GetLastEventId());

		delete preProcessor;
		delete inputStack;
	}

	TEST_METHOD(TestErrorSyntaxError)
	{
		String line;

		// init objects
		ParseContextConcrete context;
		InitInputStack(TEST_ERRORSYNTAX);
		preProcessor = new axis::application::parsing::preprocessing::PreProcessor(*inputStack, context);

		preProcessor->Prepare();

		Assert::AreEqual(true,!preProcessor->IsEOF());

		// read each line and compare
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 1")));

		Assert::AreEqual(true,context.EventSummary().HasAnyEventRegistered());

		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 2")));

		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("")));

		// do final checks
		Assert::AreEqual(true,preProcessor->IsEOF());
		Assert::AreEqual(true,preProcessor->IsSymbolDefined(_T("my_symbol")));
		Assert::AreEqual(true,context.EventSummary().HasAnyEventRegistered());
		Assert::AreEqual(4, (int)context.EventSummary().GetTotalEventCount());
		Assert::AreEqual(AXIS_ERROR_ID_DIRECTIVE_SYNTAX_ERROR, (int)context.EventSummary().GetLastEventId());

		delete preProcessor;
		delete inputStack;
	}

	TEST_METHOD(TestErrorFileNotFound)
	{
		String line;

		// init objects
		ParseContextConcrete context;
		InitInputStack(TEST_ERRORNOFILE);
		preProcessor = new axis::application::parsing::preprocessing::PreProcessor(*inputStack, context);

		preProcessor->Prepare();

		Assert::AreEqual(true,!preProcessor->IsEOF());

		// read each line and compare
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 1")));

		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("")));

		// do final checks
		Assert::AreEqual(true,preProcessor->IsEOF());
		Assert::AreEqual(true,context.EventSummary().HasAnyEventRegistered());
		Assert::AreEqual(1, (int)context.EventSummary().GetTotalEventCount());
		Assert::AreEqual(AXIS_ERROR_ID_INCLUDE_FILE_NOT_FOUND, (int)context.EventSummary().GetLastEventId());

		delete preProcessor;
		delete inputStack;
	}

	TEST_METHOD(TestErrorNoEndComment)
	{
		String line;

		// init objects
		ParseContextConcrete context;
		InitInputStack(TEST_ERRORNOENDCOMMENT);
		preProcessor = new axis::application::parsing::preprocessing::PreProcessor(*inputStack, context);

		preProcessor->Prepare();

		// at this point, the stream formatter must already have read the comments
		// line (look-ahead read), processed the error and have set the EOF flag
		Assert::AreEqual(true,preProcessor->IsEOF());

		// do final checks
		Assert::AreEqual(true,context.EventSummary().HasAnyEventRegistered());
		Assert::AreEqual(1, (int)context.EventSummary().GetTotalEventCount());
		Assert::AreEqual(AXIS_ERROR_ID_UNCLOSED_COMMENT_BLOCK, (int)context.EventSummary().GetLastEventId());

		delete preProcessor;
		delete inputStack;
	}

	TEST_METHOD(TestErrorIfErrors)
	{
		String line;

		// init objects
		ParseContextConcrete context;
		InitInputStack(TEST_ERRORIFDIRECTIVES);
		preProcessor = new axis::application::parsing::preprocessing::PreProcessor(*inputStack, context);

		preProcessor->Prepare();

		Assert::AreEqual(true,!preProcessor->IsEOF());

		// read each line and compare
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 1")));

		// check for the first error
		Assert::AreEqual(true,context.EventSummary().HasAnyEventRegistered());
		Assert::AreEqual(1, (int)context.EventSummary().GetTotalEventCount());
		Assert::AreEqual(AXIS_ERROR_ID_ELSE_WITHOUT_IF, (int)context.EventSummary().GetLastEventId());

		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 2")));
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 3")));

		// check for the second error
		Assert::AreEqual(true,context.EventSummary().HasAnyEventRegistered());
		Assert::AreEqual(2, (int)context.EventSummary().GetTotalEventCount());
		Assert::AreEqual(AXIS_ERROR_ID_ENDIF_IN_EXCESS, (int)context.EventSummary().GetLastEventId());

		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("")));

		// do final checks
		Assert::AreEqual(true,preProcessor->IsEOF());
		Assert::AreEqual(true,context.EventSummary().HasAnyEventRegistered());
		Assert::AreEqual(3, (int)context.EventSummary().GetTotalEventCount());
		Assert::AreEqual(AXIS_ERROR_ID_ENDIF_MISSING, (int)context.EventSummary().GetLastEventId());

		delete preProcessor;
		delete inputStack;
	}

	TEST_METHOD(TestErrorIOError)
	{
		String line;

		// init objects
		ParseContextConcrete context;
		char_type *buf = StringEncoding::ASCIIToUnicode(TEST_ERRORIO);
		String fileName = buf;
		delete buf;
		BogusFileReader *bogusFile = new BogusFileReader(fileName);
		inputStack = new axis::application::parsing::preprocessing::InputStack();
		((axis::application::parsing::preprocessing::InputStack*)(inputStack))->AddStream(*bogusFile);
		preProcessor = new axis::application::parsing::preprocessing::PreProcessor(*inputStack, context);

		preProcessor->Prepare();

		Assert::AreEqual(true,!preProcessor->IsEOF());

		// read lines
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("BOGUS LINE")));

		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("BOGUS LINE")));

		// check for the error
		Assert::AreEqual(true,preProcessor->IsEOF());
		Assert::AreEqual(true,context.EventSummary().HasAnyEventRegistered());
		Assert::AreEqual(1, (int)context.EventSummary().GetTotalEventCount());
		Assert::AreEqual(AXIS_ERROR_ID_INCLUDE_FILE_IO_ERROR, (int)context.EventSummary().GetLastEventId());

		delete preProcessor;
		delete inputStack;
	}

	TEST_METHOD(TestErrorManyNestedFiles)
	{
		String line;

		// init objects
		ParseContextConcrete context;
		InitInputStack(TEST_ERRORMANYNESTED01);
		preProcessor = new axis::application::parsing::preprocessing::PreProcessor(*inputStack, context);
		inputStack->SetMaximumSize(2);	// shorten max stack length to generate stack overflow error

		preProcessor->Prepare();

		// read lines
		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 1")));

		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("LINE 3")));

		line = preProcessor->ReadLine();
		Logger::WriteMessage(line.data());
		Assert::AreEqual(0, line.compare(_T("")));

		// do final checks
		Assert::AreEqual(true,preProcessor->IsEOF());
		Assert::AreEqual(true,context.EventSummary().HasAnyEventRegistered());
		Assert::AreEqual(1, (int)context.EventSummary().GetTotalEventCount());
		Assert::AreEqual(AXIS_ERROR_ID_INPUT_STACK_OVERFLOW, (int)context.EventSummary().GetLastEventId());

		delete preProcessor;
		delete inputStack;
	}

};



std::string PreProcessorTest::_testFileBasePath;
axis::services::io::FileReader *PreProcessorTest::file = NULL;
axis::application::parsing::preprocessing::PreProcessor *PreProcessorTest::preProcessor = NULL;
axis::application::parsing::preprocessing::InputStack *PreProcessorTest::inputStack = NULL;




} } }

#endif

