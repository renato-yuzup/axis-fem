#include "stdafx.h"
#include "CppUnitTest.h"
#include "CppUnitTestAssert.h"
#include "System.hpp"
#include "AxisString.hpp"
#include <sys/stat.h>
#include <direct.h>
#include <fstream>
#include "services/configuration/XmlConfigurationScript.hpp"
#include "foundation/ConfigurationNotFoundException.hpp"
#include "foundation/IOException.hpp"
#include "foundation/ElementNotFoundException.hpp"
#include "services/configuration/ScriptFactory.hpp"

#define MAX_PATH		255

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

using namespace axis;
using namespace axis::foundation;
using namespace axis::services::configuration;


#ifdef _UNICODE
	typedef std::wostringstream StringBuf;
	typedef std::wofstream FileStream;
#else
	typedef std::ostringstream StringBuf;
	typedef std::ofstream FileStream;
#endif


// These specializations are required for the use of equality asserts in 
// the MS Unit Test Framework.
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const String& s)
{
	return std::wstring(s.data());
}


namespace axis_common_library_unit_tests
{
	// This fixture tests behaviors of exceptions as established in axis Common Library component.
	TEST_CLASS(XmlConfigurationScriptTestFixture)
	{
	private:
		static std::string _xmlPath;

		static bool FileExists(std::string filename)
		{
			struct stat fileInfo;

			return (stat(filename.c_str(), &fileInfo) == 0);
		}

		static std::string GetXmlFileLocation(std::string fileName)
		{
			if (_xmlPath.empty())
			{
				char *fullPath = new char[MAX_PATH];
				_getcwd(fullPath, MAX_PATH);
				_xmlPath = fullPath;
				delete fullPath;
			}

			std::string buf = _xmlPath;
			if (buf[buf.size() - 1] != '\\' && buf[buf.size() - 1] != '/')
			{
				buf.append("/");	// acceptable on Windows
			}
			buf.append(fileName);
			return buf;
		}

		static std::string GetGoodXmlFileName(void)
		{
			return GetXmlFileLocation("test_good.xml");
		}
		static std::string GetBadXmlFileName(void)
		{
			return GetXmlFileLocation("test_bad.xml");
		}
		static std::string GetEmptyXmlFileName(void)
		{
			return GetXmlFileLocation("test_empty.xml");
		}
		static std::string GetInexistentXmlFileName(void)
		{
			return GetXmlFileLocation("test_inexistent.xml");
		}
	public:
		TEST_METHOD_INITIALIZE(SetUp)
		{
      axis::System::Initialize();
			String goodFile;
			String badFile;
			String emptyFile;
			FileStream fs;

			// check file existence
			if (FileExists(GetGoodXmlFileName()))
			{	// fail set-up
				String ws;
				std::string s("Cannot create XML test file ");
				s.append(GetGoodXmlFileName()).append(" -- file already exists.");
				StringEncoding::AssignFromASCII(s.data(), ws);
				Assert::Fail(ws.data());
			}
			if (FileExists(GetBadXmlFileName()))
			{	// fail set-up
				String ws;
				std::string s("Cannot create XML test file ");
				s.append(GetBadXmlFileName()).append(" -- file already exists.");
				StringEncoding::AssignFromASCII(s.data(), ws);
				Assert::Fail(ws.data());
			}
			if (FileExists(GetEmptyXmlFileName()))
			{	// fail set-up
				String ws;
				std::string s("Cannot create XML test file ");
				s.append(GetEmptyXmlFileName()).append(" -- file already exists.");
				StringEncoding::AssignFromASCII(s.data(), ws);
				Assert::Fail(ws.data());
			}
			if (FileExists(GetInexistentXmlFileName()))
			{	// fail set-up
				String ws;
				std::string s("The XML file ");
				s.append(GetInexistentXmlFileName()).append(" already exists. Please delete this file or else tests may fail.");
				StringEncoding::AssignFromASCII(s.data(), ws);
				Assert::Fail(ws.data());
			}

			StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), goodFile);
			StringEncoding::AssignFromASCII(GetBadXmlFileName().data(), badFile);
			StringEncoding::AssignFromASCII(GetEmptyXmlFileName().data(), emptyFile);

			// create well-formed XML test file
			StringBuf contents;
			contents <<  _T("<system>") << std::endl;
			contents <<  _T("  <section1 id=\"A\">") << std::endl;
			contents <<  _T("    <test>Test1</test>") << std::endl;
			contents <<  _T("    <test-1 attr1=\"B\" attr2=\"C\" />") << std::endl;
			contents <<  _T("    <test attr1=\"F\">") << std::endl;
			contents <<  _T("      TESTTEST") << std::endl;
			contents <<  _T("    </test>") << std::endl;
			contents <<  _T("  </section1>") << std::endl;
			contents <<  _T("  <section2>") << std::endl;
			contents <<  _T("    <test name=\"HELLO\" />") << std::endl;
			contents <<  _T("    <test name=\"WORLD\" />") << std::endl;
			contents <<  _T("  </section2>") << std::endl;
			contents <<  _T("  <section3 /> <!-- This is an empty node -->") << std::endl;
			contents <<  _T("</system>") << std::endl;
			contents <<  _T("<system2>") << std::endl;
			contents <<  _T("  <!-- Here it is another empty node -->") << std::endl;
			contents <<  _T("  <test-empty />") << std::endl;
			contents <<  _T("</system2>") << std::endl;
			fs.open(goodFile.data());
			fs.write(contents.str().data(), contents.str().size());
			fs.close();

			// create bad-formed XML test file
			contents = StringBuf(); // clear buffer
			contents <<  _T("<system>") << std::endl;
			contents <<  _T("  <section1 id=\"A\">") << std::endl;
			contents <<  _T("    <test>Test1</test>") << std::endl;
			contents <<  _T("    <test-1 attr1=\"B\" attr2=\"C\" />") << std::endl;
			contents <<  _T("    <test attr1=\"F\">") << std::endl;
			contents <<  _T("      TESTTEST") << std::endl;
			contents <<  _T("    </test>") << std::endl;
			contents <<  _T("  </section1>") << std::endl;
			contents <<  _T("  <section2>") << std::endl;
			contents <<  _T("    <test name=\"HELLO\" />") << std::endl;
			contents <<  _T("    <test name=\"WORLD\" />") << std::endl;
			contents <<  _T("  </section2>") << std::endl;
			contents <<  _T("  <section3 /> <!-- This is an empty node -->") << std::endl;
			contents <<  _T("</system-1>") << std::endl; // introduced error
			contents <<  _T("<system2>") << std::endl;
			contents <<  _T("  <!-- Here it is another empty node -->") << std::endl;
			contents <<  _T("  <test-empty />") << std::endl;
			contents <<  _T("</system2>") << std::endl;
			fs.open(badFile.data());
			fs.write(contents.str().data(), contents.str().size());
			fs.close();

			// create empty XML file
			fs.open(emptyFile.data());
			fs.close();
		}

		TEST_METHOD_CLEANUP(TearDown)
		{
			// destroy XML test file
			remove(GetGoodXmlFileName().c_str());
			remove(GetBadXmlFileName().c_str());
			remove(GetEmptyXmlFileName().c_str());
      axis::System::Finalize();
		}

		TEST_METHOD(TestFileOpenAndObjectDestructor)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}
			try
			{
				// destroy file
				delete cs;
			}
			catch (...)
			{
				Assert::Fail(_T("Destructor failed!"));
			}
		}
		TEST_METHOD(TestRootElement)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			Assert::AreEqual(_T("system"), cs->GetSectionName());
		}
		TEST_METHOD(TestForEmptyFile)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetEmptyXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
				Assert::Fail(_T("Exception not triggered while reading empty file."));
			}
			catch(ConfigurationNotFoundException)
			{
				// ok, success!
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception while reading empty file."));
			}
		}
		TEST_METHOD(TestForNonExistentFile)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetInexistentXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
				Assert::Fail(_T("Exception not triggered while reading inexistent file."));
			}
			catch(IOException)
			{
				// ok, success!
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception while reading inexistent file."));
			}
		}
		TEST_METHOD(TestAnonymousSubSectionNavigation)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// verify that we are in the correct root note
			Assert::AreEqual(0, cs->GetSectionName().compare(_T("system")));

			// try to obtain any subsection
			ConfigurationScript &subsection = cs->GetFirstChildSection();
			Assert::AreEqual(true, &subsection != NULL);
			Assert::AreEqual(true, &subsection != cs);
			Assert::AreEqual(0, subsection.GetSectionName().compare(_T("section1")));
		}
		TEST_METHOD(TestForNoChildrenNavigation)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// verify that we are in the correct root note
			Assert::AreEqual(0, cs->GetSectionName().compare(_T("system")));

			// try to obtain any subsection
			ConfigurationScript &subsection = cs->GetSection(_T("section3"));
			Assert::AreEqual(true, &subsection != NULL);
			Assert::AreEqual(true, &subsection != cs);
			Assert::AreEqual(0, subsection.GetSectionName().compare(_T("section3")));
			Assert::AreEqual(true, !subsection.HasChildSections());
		}
		TEST_METHOD(TestNamedSubsectionNavigation)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// verify that we are in the correct root note
			Assert::AreEqual(0, cs->GetSectionName().compare(_T("system")));

			// try to obtain any subsection
			ConfigurationScript &subsection1 = cs->GetSection(_T("section1"));
			ConfigurationScript &subsection2 = cs->GetSection(_T("section2"));
			Assert::AreEqual(true, &subsection1 != NULL);
			Assert::AreEqual(true, &subsection1 != cs);
			Assert::AreEqual(true, &subsection2 != NULL);
			Assert::AreEqual(true, &subsection2 != cs);
			Assert::AreEqual(true, &subsection2 != &subsection1);
			Assert::AreEqual(0, subsection1.GetSectionName().compare(_T("section1")));
			Assert::AreEqual(0, subsection2.GetSectionName().compare(_T("section2")));
			Assert::AreEqual(true, subsection1.HasChildSections());
			Assert::AreEqual(true, subsection2.HasChildSections());
		}
		TEST_METHOD(TestForInexistentSubsectionNavigation)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// verify that we are in the correct root note
			Assert::AreEqual(0, cs->GetSectionName().compare(_T("system")));

			// try to obtain a bogus subsection
			try
			{
				ConfigurationScript &subsection = cs->GetSection(_T("section0"));
				Assert::Fail(_T("Expected exception not thrown when reading bogus section."));
			}
			catch (ElementNotFoundException)
			{
				// ok, success!
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown when reading bogus section."));
			}
		}
		TEST_METHOD(TestAnonymousSiblingNavigation)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// verify that we are in the correct root note
			Assert::AreEqual(0, cs->GetSectionName().compare(_T("system")));

			// obtain the first child section
			ConfigurationScript &subsection = cs->GetFirstChildSection();
			Assert::AreEqual(0, subsection.GetSectionName().compare(_T("section1")));

			// get next sibling
			ConfigurationScript &sibling = subsection.GetNextSiblingSection();
			Assert::AreEqual(0, sibling.GetSectionName().compare(_T("section2")));
		}
		TEST_METHOD(TestNamedSiblingNavigation)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// verify that we are in the correct root note
			Assert::AreEqual(0, cs->GetSectionName().compare(_T("system")));

			// obtain the first child section
			ConfigurationScript &subsection = cs->GetFirstChildSection();
			Assert::AreEqual(0, subsection.GetSectionName().compare(_T("section1")));

			ConfigurationScript *sibling = subsection.FindNextSibling(_T("section3"));
			Assert::AreEqual(true, sibling != NULL);
			Assert::AreEqual(0, sibling->GetSectionName().compare(_T("section3")));
		}
		TEST_METHOD(TestForNoMoreNamedSiblingNavigation)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// verify that we are in the correct root note
			Assert::AreEqual(0, cs->GetSectionName().compare(_T("system")));

			// obtain the first child section
			ConfigurationScript &subsection = cs->GetFirstChildSection();
			Assert::AreEqual(0, subsection.GetSectionName().compare(_T("section1")));

			ConfigurationScript *previousItem = NULL;
			ConfigurationScript *item = &subsection.GetSection(_T("test"));
			Assert::AreEqual(0, item->GetSectionName().compare(_T("test")));
			previousItem = item;
			item = item->FindNextSibling(_T("test"));
			Assert::AreEqual(true, item != previousItem);
			Assert::AreEqual(0, item->GetSectionName().compare(_T("test")));
			item = item->FindNextSibling(_T("test"));
			Assert::AreEqual(true, item == NULL);
		}
		TEST_METHOD(TestNestedNavigation)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// verify that we are in the correct root note
			Assert::AreEqual(0, cs->GetSectionName().compare(_T("system")));

			// obtain the first child section
			ConfigurationScript &subsection = cs->GetSection(_T("section2"));
			Assert::AreEqual(0, subsection.GetSectionName().compare(_T("section2")));

			// iterate through all 'test'-named child nodes
			ConfigurationScript *previousItem = NULL;
			ConfigurationScript *item = &subsection.GetSection(_T("test"));
			Assert::AreEqual(0, item->GetSectionName().compare(_T("test")));
			previousItem = item;
			item = item->FindNextSibling(_T("test"));
			Assert::AreEqual(true, item != previousItem);
			Assert::AreEqual(0, item->GetSectionName().compare(_T("test")));
			item = item->FindNextSibling(_T("test"));
			Assert::AreEqual(true, item == NULL); // no more sibling nodes with this name

			// go to another branch and test navigation
			item = &cs->GetSection(_T("section1"));
			Assert::AreEqual(0, item->GetSectionName().compare(_T("section1")));
			item = &item->GetSection(_T("test-1"));
			Assert::AreEqual(0, item->GetSectionName().compare(_T("test-1")));
		}
		TEST_METHOD(TestForNoMoreNestingNavigation)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// verify that we are in the correct root note
			Assert::AreEqual(0, cs->GetSectionName().compare(_T("system")));

			// obtain the second child section
			ConfigurationScript &subsection = cs->GetSection(_T("section2"));
			Assert::AreEqual(0, subsection.GetSectionName().compare(_T("section2")));

			// iterate through children nodes
			ConfigurationScript *item = &subsection.GetSection(_T("test"));
			Assert::AreEqual(0, item->GetSectionName().compare(_T("test")));
			Assert::AreEqual(true, !item->HasChildSections());

			// try to get a new child; must throw exception
			try
			{
				item->GetFirstChildSection();

				// huh?
				Assert::Fail(_T("Expected exception was not thrown when obtaining an inexistent child node."));
			}
			catch (ElementNotFoundException)
			{
				// ok, success!
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception was thrown when obtaining an inexistent child node."));
			}

			// try to change to a new root node -- must fail because we only
			// accept XML files with only one root node
			try
			{
				item = &cs->GetSection(_T("system2"));
				Assert::Fail(_T("Expected exception was not thrown when moving to a new root node."));
			}
			catch (ElementNotFoundException)
			{
				// ok, success!
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception was thrown when moving to a new root node."));
			}

			// last test: move to the last sibling in a branch...
			item = &cs->GetSection(_T("section2")).GetFirstChildSection().GetNextSiblingSection();
			Assert::AreEqual(true, !item->HasMoreSiblingsSection());

			// ...and try to get a new sibling (must fail)
			try
			{
				item->GetNextSiblingSection();

				// huh?
				Assert::Fail(_T("Expected exception was not thrown when obtaining an inexistent sibling node."));
			}
			catch (ElementNotFoundException)
			{
				// ok, success!
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception was thrown when obtaining an inexistent sibling node."));
			}
		}
		TEST_METHOD(TestAttributeRead)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// verify that we are in the correct root note
			Assert::AreEqual(0, cs->GetSectionName().compare(_T("system")));

			// obtain the second child section
			ConfigurationScript &subsection = cs->GetSection(_T("section2"));
			Assert::AreEqual(0, subsection.GetSectionName().compare(_T("section2")));
			ConfigurationScript *item = &subsection.GetSection(_T("test"));

			// get attributes
			Assert::AreEqual(true, item->ContainsAttribute(_T("name")));
			Logger::WriteMessage(item->GetAttributeValue(_T("name")));
			Assert::AreEqual(0, item->GetAttributeValue(_T("name")).compare(_T("HELLO")));

			// go to another branch and test
			item = &cs->GetSection(_T("section1")).GetSection(_T("test")).GetNextSiblingSection();
			Assert::AreEqual(0, item->GetSectionName().compare(_T("test-1")));
			Assert::AreEqual(true, item->ContainsAttribute(_T("attr1")));
			Assert::AreEqual(0, item->GetAttributeValue(_T("attr1")).compare(_T("B")));
			Assert::AreEqual(true, item->ContainsAttribute(_T("attr2")));
			Assert::AreEqual(0, item->GetAttributeValue(_T("attr2")).compare(_T("C")));
		}
		TEST_METHOD(TestAttributeReadWithDefault)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// go to a note with an existent attribute (system > section1 > test [second child])
			ConfigurationScript &subsection = cs->GetSection(_T("section1")).GetFirstChildSection().GetNextSiblingSection();

			// try to get current value
			Assert::AreEqual(0, subsection.GetAttributeWithDefault(_T("attr1"), _T("Default-Value")).compare(_T("B")));

			// try to read an inexistent attribute
			Assert::AreEqual(0, subsection.GetAttributeWithDefault(_T("bogus-attr"), _T("Default-Value")).compare(_T("Default-Value")));
			Assert::AreEqual(true, !subsection.ContainsAttribute(_T("bogus-attr")));
		}
		TEST_METHOD(TestFailingAttributeRead)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// go to a note with attributes (system > section1 > test-1)
			ConfigurationScript &subsection = cs->GetSection(_T("section1")).GetFirstChildSection().GetNextSiblingSection();

			// try to read an inexistent attribute
			try
			{
				subsection.GetAttributeValue(_T("bogus-attr"));

				Assert::Fail(_T("Expected exception was not thrown."));
			}
			catch (ElementNotFoundException)
			{
				// ok, success!
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception was thrown."));
			}
		}
		TEST_METHOD(TestTextRead)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// go to a note with text value (system > section1 > test [third child])
			ConfigurationScript &subsection = cs->GetSection(_T("section1"))
				.GetFirstChildSection()
				.GetNextSiblingSection()
				.GetNextSiblingSection();

			// get test text data
			Assert::AreEqual(0, subsection.GetConfigurationText().compare(_T("TESTTEST")));
		}
		TEST_METHOD(TestEmptyTextRead)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// go to a note without text value (system > section1)
			ConfigurationScript &subsection = cs->GetSection(_T("section1"));

			// test node text data
			Assert::AreEqual(true, subsection.GetConfigurationText().empty());
		}
		TEST_METHOD(TestSingleInstanceObjects)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// get some node...
			ConfigurationScript *section1 = &cs->GetSection(_T("section1"));
			Assert::AreEqual(0, section1->GetSectionName().compare(_T("section1")));

			ConfigurationScript *firstItem = &section1->GetFirstChildSection();
			Assert::AreEqual(0, firstItem->GetSectionName().compare(_T("test")));

			// try to retrieve these same nodes using both the same operations or equivalent ones
			ConfigurationScript *node = &cs->GetSection(_T("section1"));
			Assert::AreEqual(true, node == section1);
			node = &node->GetSection(_T("test"));
			Assert::AreEqual(true, node == firstItem);
		}
		TEST_METHOD(TestConfigurationPath)
		{
			ConfigurationScript *cs;
			try
			{
				// create object and read from existent file
				String buf;	StringEncoding::AssignFromASCII(GetGoodXmlFileName().data(), buf);
				cs = &ScriptFactory::ReadFromXmlFile(buf);
			}
			catch (...)
			{
				Assert::Fail(_T("Constructor failed!"));
			}

			// get some node...
			String expectedPath = _T("system.section1.test<2>");
			ConfigurationScript *testNode = &cs->GetSection(_T("section1"))
				.GetFirstChildSection()
				.GetNextSiblingSection()
				.GetNextSiblingSection();
			Assert::AreEqual(0, testNode->GetConfigurationPath().compare(expectedPath));

			// get other node...
			expectedPath = _T("system.section1.test<1>");
			testNode = &cs->GetSection(_T("section1"))
				.GetFirstChildSection();
			Assert::AreEqual(0, testNode->GetConfigurationPath().compare(expectedPath));
		}
	};

}


std::string axis_common_library_unit_tests::XmlConfigurationScriptTestFixture::_xmlPath;