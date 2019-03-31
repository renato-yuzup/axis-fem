#if defined DEBUG || defined _DEBUG

#include "unit_tests/unit_tests.hpp"

#include "System.hpp"
#include "AxisString.hpp"
#include "services/logging/LogFile.hpp"
#include "application/runnable/AxisApplication.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "application/output/ResultBucket.hpp"
#include "application/runnable/AxisApplication.hpp"
#include "domain/elements/Node.hpp"
#undef GetFileTitle
#include "services/io/FileSystem.hpp"
#include "services/io/FileWriter.hpp"
#include "services/logging/LogFile.hpp"
#include "application/jobs/JobRequest.hpp"
#include "services/management/ServiceLocator.hpp"
#include "application/factories/parsers/BlockProvider.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/language/syntax/evaluation/IdValue.hpp"
#include "services/language/syntax/evaluation/ArrayValue.hpp"
#include "application/jobs/AnalysisStep.hpp"
#include "application/jobs/WorkFolder.hpp"

using namespace axis;
using namespace axis::foundation;
using namespace axis::services::logging;
using namespace axis::services::io;
using namespace axis::domain::analyses;
using namespace axis::domain;
using namespace axis::domain::elements;
using namespace axis::application::runnable;
using namespace axis::application::jobs;
using namespace axis::services::management;
using namespace axis::application::factories::parsers;
using namespace axis::application::parsing::core;
using namespace axis::application::parsing::parsers;
using namespace axis::services::language::syntax::evaluation;

namespace {
const String::char_type * ConfigFileLocation = _T("test_settings.config");
const String::char_type * InputFileLocation = _T("test_input_master_file.axis");
const String::char_type * BaseIncludeFolderLocation = _T(".");
const String::char_type * OutputFolderLocation = _T(".");
const String::char_type * OutputLogLocation = _T("test_output.log");
const String::char_type * AppLogLocation = _T("test_app.log");
const String::char_type * baseSamplesDirLocation = _T("Model Input Files/cantilever_beam_20k");
}

namespace axis { namespace unit_tests { namespace core {

	/* ================================================================================================================== */
	/* ============================================= OUR TEST FIXTURE CLASS ============================================= */
	TEST_CLASS(AxisApplicationTest)
	{
	private:
		/* =============== AUXILIARY METHODS =============== */
		static String GetFullPath(const String& filename)
		{
			return FileSystem::ConcatenatePath(FileSystem::GetApplicationFolder(), filename);
		}
		static void CreateConfigTestFile(const axis::String& filename)
		{
			FileWriter& writer = FileWriter::Create(filename);
			writer.Open(StreamWriter::kOverwrite, StreamWriter::kExclusiveMode);

			writer.WriteLine(_T("<axis.settings>"));
			writer.WriteLine(_T("  <plugins>"));
			writer.WriteLine(_T("    <system_plugins>"));
			writer.WriteLine(_T("      <plugin location = \"axis.StandardElements.dll\" />"));
			writer.WriteLine(_T("      <plugin location = \"axis.StandardMaterials.dll\" />"));
      writer.WriteLine(_T("      <plugin location = \"axis.Echinopsis.dll\" />"));
      writer.WriteLine(_T("      <plugin location = \"axis.Orange.dll\" />"));
      writer.WriteLine(_T("      <plugin location = \"axis.Solver.dll\" />"));
			writer.WriteLine(_T("    </system_plugins>"));
			writer.WriteLine(_T("    <stable_plugins />"));
			writer.WriteLine(_T("    <volatile_plugins />"));
			writer.WriteLine(_T("  </plugins>"));
			writer.WriteLine(_T("</axis.settings>"));

			writer.Close();
		}
		static void BuildInputFileWithCrossRef(const axis::String& filename)
		{
			FileWriter& writer = FileWriter::Create(filename);
			writer.Open(StreamWriter::kOverwrite, StreamWriter::kExclusiveMode);

			writer.WriteLine(_T("######################################### ANALYSIS INPUT FILE #########################################"));
			writer.WriteLine(_T("## "));
			writer.WriteLine(_T("## This file describes the input model and configures the solver for the numerical analysis."));
			writer.WriteLine(_T("## Please note that the interpreter is case-senstive but not position-sensitive, in contrast to"));
			writer.WriteLine(_T("## some commercial solvers."));
			writer.WriteLine(_T("## "));
			writer.WriteLine(_T("#######################################################################################################"));
			writer.WriteLine(_T("# "));
			writer.WriteLine(_T("# "));
			writer.WriteLine(_T("# "));
			writer.WriteLine(_T("/*"));
			writer.WriteLine(_T("  -- ANALYSIS BLOCK"));
			writer.WriteLine(_T("  This section describes time range to analyze, solvers to use, how to collect"));
			writer.WriteLine(_T("  results and boundary conditions applied to the model in a step basis. */"));
			writer.WriteLine(_T("BEGIN ANALYSIS"));
			writer.WriteLine(_T("   /*"));
			writer.WriteLine(_T("     -- STEP BLOCK"));
			writer.WriteLine(_T("     Each step block defines a contiguous time range in which a specific solver is"));
			writer.WriteLine(_T("     used to analyze the phenomenom. */"));
			writer.WriteLine(_T("   BEGIN STEP WITH TYPE = LINEAR_STATIC, START_TIME = 0, END_TIME = 0.25"));
			writer.WriteLine(_T("      /*"));
			writer.WriteLine(_T("        -- SNAPSHOTS BLOCK"));
			writer.WriteLine(_T("        A snapshot block defines when result should be collected from the model. */"));
			writer.WriteLine(_T("      BEGIN SNAPSHOTS"));
			writer.WriteLine(_T("         SNAPSHOT AT 0.25"));
			writer.WriteLine(_T("      END SNAPSHOTS"));
			writer.WriteLine(_T("      /*"));
			writer.WriteLine(_T("        -- OUTPUT BLOCK"));
			writer.WriteLine(_T("        Each output block defines an output file. The statements inside it declares the"));
			writer.WriteLine(_T("        information that should be written on it. */"));
			writer.WriteLine(_T("      BEGIN OUTPUT WITH FILE=\"results\", FORMAT = REPORT"));
			writer.WriteLine(_T("         RECORD NODAL DISPLACEMENT ALL ON SET all_nodes"));
			writer.WriteLine(_T("      END OUTPUT"));
			writer.WriteLine(_T("      /*"));
			writer.WriteLine(_T("        -- CONSTRAINTS BLOCK"));
			writer.WriteLine(_T("        This section describes any restriction applied to nodes or elements."));
			writer.WriteLine(_T("        Restriction is any movement, velocity or acceleration imposed to the"));
			writer.WriteLine(_T("        object. */"));
			writer.WriteLine(_T("      BEGIN CONSTRAINTS"));
			writer.WriteLine(_T("         LOCK SET engaste IN ALL DIRECTIONS"));
			writer.WriteLine(_T("      END CONSTRAINTS"));
			writer.WriteLine(_T("      /*"));
			writer.WriteLine(_T("        -- LOADS BLOCK"));
			writer.WriteLine(_T("        This section describes loads applied to the model. Only sub-blocks"));
			writer.WriteLine(_T("        describing loads should be declared here. Orphaned declarations "));
			writer.WriteLine(_T("        will be treated as an invalid syntax. */"));
			writer.WriteLine(_T("      BEGIN LOADS"));
			writer.WriteLine(_T("         BEGIN NODAL_LOADS"));
			writer.WriteLine(_T("            ON SET nodal_load_set BEHAVES AS \"constant load\" ON Y DIRECTION"));
			writer.WriteLine(_T("         END NODAL_LOADS"));
			writer.WriteLine(_T("      END LOADS"));
			writer.WriteLine(_T("   END STEP"));
			writer.WriteLine(_T("END ANALYSIS"));
			writer.WriteLine();
			writer.WriteLine();
			writer.WriteLine(_T("/*"));
			writer.WriteLine(_T("  -- ELEMENTS BLOCK"));
			writer.WriteLine(_T("  This section describes model elements that share the same element type and material."));
			writer.WriteLine(_T("  It is required that each element block have an associated part declaration (see the"));
			writer.WriteLine(_T("  PARTS block for more information).  */"));
			writer.WriteLine(_T("BEGIN ELEMENTS WITH SET_ID = \"all_elements\""));
			writer.WriteLine(_T("  # ----------------------------------------------------------------"));
			writer.WriteLine(_T("  # ELEM. ID |  CONNECTIVITY LIST"));
			writer.WriteLine(_T("  # ----------------------------------------------------------------"));
			writer.WriteLine(_T("           1 :   1   2   3   4   7   8   9  10      "));
			writer.WriteLine(_T("           2 :   4   3   5   6  10   9  11  12      "));
			writer.WriteLine(_T("  # ----------------------------------------------------------------"));
			writer.WriteLine(_T("END ELEMENTS"));
			writer.WriteLine();
			writer.WriteLine();
			writer.WriteLine(_T("/*"));
			writer.WriteLine(_T("  -- NODE_SET BLOCK"));
			writer.WriteLine(_T("  This section describes a node set, which can be used to apply boundary"));
			writer.WriteLine(_T("  conditions or select specific nodes for information output. * One block"));
			writer.WriteLine(_T("  describes one node set.*/"));
			writer.WriteLine(_T("BEGIN NODE_SET WITH ID = nodal_load_set"));
			writer.WriteLine(_T("   5  6"));
			writer.WriteLine(_T("END NODE_SET"));
			writer.WriteLine();
			writer.WriteLine();
			writer.WriteLine(_T("BEGIN NODE_SET WITH ID = engaste"));
			writer.WriteLine(_T("   1-2"));
			writer.WriteLine(_T("   7"));
			writer.WriteLine(_T("   8"));
			writer.WriteLine(_T("END NODE_SET"));
			writer.WriteLine();
			writer.WriteLine();
			writer.WriteLine(_T("# -- PARTS BLOCK"));
			writer.WriteLine(_T("# This section determines the element type (and all required attributes) and the"));
			writer.WriteLine(_T("# material type on a per element set basis."));
			writer.WriteLine(_T("BEGIN PARTS WITH ELEM_TYPE=LINEAR_HEXAHEDRON,     # use linear hexahedron"));
			writer.WriteLine(_T("                 PROPERTIES=()                    # no required properties"));
			writer.WriteLine();
			writer.WriteLine(_T("    # This will make all elements in the 'all_elements' set use a linear isotropic elastic material model."));
 			writer.WriteLine(_T("    SET all_elements IS LINEAR_ISO_ELASTIC WITH POISSON         = 0.3   ,"));
 			writer.WriteLine(_T("                                                ELASTIC_MODULUS = 200E9,"));
			writer.WriteLine(_T("                                                LINEAR_DENSITY = 7850"));
			writer.WriteLine(_T("END PARTS  # The name must match with the one on the corresponding heading"));
			writer.WriteLine();
			writer.WriteLine();
			writer.WriteLine(_T(""));
			writer.WriteLine();
			writer.WriteLine(_T("/*"));
			writer.WriteLine(_T("  -- CURVES BLOCK"));
			writer.WriteLine(_T("  This section describes curves to describe behavior of loads along time."));
			writer.WriteLine(_T("  Only sub-blocks representing curves should be declared here. Orphaned"));
			writer.WriteLine(_T("  declarations will be treated as an invalid syntax. */"));
			writer.WriteLine(_T("BEGIN CURVES"));
			writer.WriteLine(_T("   BEGIN MULTILINE_CURVE WITH ID=\"constant load\""));
			writer.WriteLine(_T("      # --------------------"));
			writer.WriteLine(_T("      #    x     |    y     "));
			writer.WriteLine(_T("      # --------------------"));
			writer.WriteLine(_T("           0         -100   "));
			writer.WriteLine(_T("           1         -100   "));
			writer.WriteLine(_T("      # --------------------"));
			writer.WriteLine(_T("   END MULTILINE_CURVE"));
			writer.WriteLine(_T("END CURVES"));
			writer.WriteLine();
			writer.WriteLine();
			writer.WriteLine(_T("/*"));
			writer.WriteLine(_T("  -- NODES BLOCK"));
			writer.WriteLine(_T("  This section describes the nodes of the model and optionally includes"));
			writer.WriteLine(_T("  them in a node set. It is not mandatory that all nodes must be declared"));
			writer.WriteLine(_T("  at once. Nodes can be separated in several blocks as seen convenient.  */"));
			writer.WriteLine(_T("BEGIN NODES WITH SET_ID = \"all_nodes\""));
			writer.WriteLine(_T("  # --------------------------------"));
			writer.WriteLine(_T("  # NODE ID |   X   |   Y   |  Z    "));
			writer.WriteLine(_T("  # --------------------------------"));
			writer.WriteLine(_T("          1:    0   ,   0   ,  0    "));
			writer.WriteLine(_T("          2:    0   ,   0   ,  1    "));
			writer.WriteLine(_T("          3:    1.  ,   0   ,  1.   "));
			writer.WriteLine(_T("          9:    1   ,   1   ,  1    "));
			writer.WriteLine(_T("         10:    1   ,   1   ,  0    "));
			writer.WriteLine(_T("         11:    2   ,   1   ,  1    "));
			writer.WriteLine(_T("         12:    2   ,   1   ,  0    "));
			writer.WriteLine(_T("          4:    1.  ,   0   ,  0    "));
			writer.WriteLine(_T("          5:   2.0  ,   0   , 1.0   "));
			writer.WriteLine(_T("          6:   2.0  ,   0   ,  0    "));
			writer.WriteLine(_T("          7:    0   ,   1   ,  0    "));
			writer.WriteLine(_T("          8:    0   ,   1   ,  1    "));
			writer.WriteLine(_T("  # --------------------------------"));
			writer.WriteLine(_T("END NODES"));
			writer.WriteLine();
			writer.WriteLine();

			writer.Close();
		}
		static void CreateModelTestFile(const axis::String& filename)
		{
			BuildInputFileWithCrossRef(filename);
		}
		static void DeleteConfigTestFile(const axis::String& filename)
		{

		}
		static void DeleteModelTestFile(const axis::String& filename)
		{

		}
	public:
		TEST_METHOD_INITIALIZE(SetUp)
		{
      axis::System::Initialize();
			CreateConfigTestFile(GetFullPath(ConfigFileLocation));
			CreateModelTestFile(GetFullPath(InputFileLocation));
		}

		TEST_METHOD_CLEANUP(TearDown)
		{
			DeleteConfigTestFile(GetFullPath(ConfigFileLocation));
			DeleteModelTestFile(GetFullPath(InputFileLocation));
      axis::System::Finalize();
		}

		TEST_METHOD(TestConstructor)
		{
			AxisApplication& app = AxisApplication::CreateApplication();
			app.Destroy();
		}

		TEST_METHOD(TestBootstrap)
		{
			AxisApplication& app = AxisApplication::CreateApplication();

			app.Configuration().SetConfigurationScriptPath(FileSystem::ConcatenatePath(
        FileSystem::GetApplicationFolder(), ConfigFileLocation));
			app.Bootstrap();
			Assert::AreEqual(true, app.IsSystemReady());

			// check that our external plugins have been loaded
			// successfully; as plugins are loaded in the order it appears
			// in the configuration file, the enumeration will happen at
			// exact the same order
			Assert::AreEqual(5L, app.GetPluginLinkCount());
      String echinopsisPluginName = app.GetPluginLinkInformation(0).GetPluginPath();
      String essentialsPluginName = app.GetPluginLinkInformation(1).GetPluginPath();
      String solverPluginName = app.GetPluginLinkInformation(2).GetPluginPath();
			String stdElementsPluginName = app.GetPluginLinkInformation(3).GetPluginPath();
			String stdMaterialsPluginName = app.GetPluginLinkInformation(4).GetPluginPath();
      echinopsisPluginName = FileSystem::GetFileTitle(echinopsisPluginName);
			stdElementsPluginName = FileSystem::GetFileTitle(stdElementsPluginName);
			stdMaterialsPluginName = FileSystem::GetFileTitle(stdMaterialsPluginName);
      essentialsPluginName = FileSystem::GetFileTitle(essentialsPluginName);
      solverPluginName = FileSystem::GetFileTitle(solverPluginName);
      Assert::AreEqual(_T("axis.echinopsis"), echinopsisPluginName.to_lower_case());
			Assert::AreEqual(_T("axis.standardelements"), stdElementsPluginName.to_lower_case());
			Assert::AreEqual(_T("axis.standardmaterials"), stdMaterialsPluginName.to_lower_case());
      Assert::AreEqual(_T("axis.orange"), essentialsPluginName.to_lower_case());
      Assert::AreEqual(_T("axis.solver"), solverPluginName.to_lower_case());

			app.Destroy();
		}

		TEST_METHOD(TestInputParsersNesting)
		{
			// initialize application
			AxisApplication& app = AxisApplication::CreateApplication();
			app.Configuration().SetConfigurationScriptPath(FileSystem::ConcatenatePath(
        FileSystem::GetApplicationFolder(), ConfigFileLocation));
			app.Bootstrap();

			// get module manager and verify existence of root input 
			// parser provider
			const GlobalProviderCatalog& manager = app.GetModuleManager();
			Assert::AreEqual(true, manager.ExistsProvider(
        ServiceLocator::GetMasterInputParserProviderPath()));
			BlockProvider& rootInputProvider = 
        static_cast<BlockProvider&>(manager.GetProvider(
        ServiceLocator::GetMasterInputParserProviderPath()));

			// create some parameter lists we will need
			ParameterList& idParamList = ParameterList::Create();
			idParamList.AddParameter(_T("ID"), *new IdValue(_T("test")));

			ParameterList& setIdParamList = ParameterList::Create();
			setIdParamList.AddParameter(_T("SET_ID"), *new IdValue(_T("test")));

			ParameterList& partsParamList = ParameterList::Create();
			partsParamList.AddParameter(_T("ELEM_TYPE"), *new IdValue(_T("LINEAR_HEXAHEDRON")));
			partsParamList.AddParameter(_T("PROPERTIES"), *new ArrayValue());

			// check existence of each of the possible blocks residing 
			// directly under the root of the analysis file
			Assert::AreEqual(true, rootInputProvider.ContainsProvider(_T("ANALYSIS"), ParameterList::Empty));
			Assert::AreEqual(true, rootInputProvider.ContainsProvider(_T("NODES"), setIdParamList));
			Assert::AreEqual(true, rootInputProvider.ContainsProvider(_T("ELEMENTS"), setIdParamList));
			Assert::AreEqual(true, rootInputProvider.ContainsProvider(_T("NODE_SET"), idParamList));
			Assert::AreEqual(true, rootInputProvider.ContainsProvider(_T("ELEMENT_SET"), idParamList));
			Assert::AreEqual(true, rootInputProvider.ContainsProvider(_T("PARTS"), partsParamList));
			Assert::AreEqual(true, rootInputProvider.ContainsProvider(_T("CURVES"), ParameterList::Empty));

			app.Destroy();
		}

		TEST_METHOD(TestReadAnalysis)
		{
			AxisApplication& app = AxisApplication::CreateApplication();

			// start application and output log
			LogFile& appLog = LogFile::Create(AppLogLocation);
			LogFile& analysisLog = LogFile::Create(OutputLogLocation);
			app.ConnectListener(appLog);
			appLog.StartLogging();
			analysisLog.StartLogging();

			app.Configuration().SetConfigurationScriptPath(FileSystem::ConcatenatePath(
        FileSystem::GetApplicationFolder(), ConfigFileLocation));
			app.Bootstrap();
			Assert::AreEqual(true, app.IsSystemReady());

			app.ConnectListener(analysisLog);
			JobRequest job(InputFileLocation, BaseIncludeFolderLocation, _T("."));
			app.SubmitJob(job);
			const StructuralAnalysis& analysis = app.GetJobWorkspace();
			const NumericalModel& model = analysis.GetNumericalModel();

			// stop logging
			appLog.StopLogging();
			analysisLog.StopLogging();

			// validate general characteristics of the analysis
			Assert::AreEqual(12, (int)model.Nodes().Count());
			Assert::AreEqual(2, (int)model.Elements().Count());
			Assert::AreEqual(1, (int)model.Curves().Count());

			// check that every node exists
			Assert::AreEqual(true, model.Nodes().IsUserIndexed(1));
			Assert::AreEqual(true, model.Nodes().IsUserIndexed(2));
			Assert::AreEqual(true, model.Nodes().IsUserIndexed(3));
			Assert::AreEqual(true, model.Nodes().IsUserIndexed(4));
			Assert::AreEqual(true, model.Nodes().IsUserIndexed(5));
			Assert::AreEqual(true, model.Nodes().IsUserIndexed(6));
			Assert::AreEqual(true, model.Nodes().IsUserIndexed(7));
			Assert::AreEqual(true, model.Nodes().IsUserIndexed(8));
			Assert::AreEqual(true, model.Nodes().IsUserIndexed(9));
			Assert::AreEqual(true, model.Nodes().IsUserIndexed(10));
			Assert::AreEqual(true, model.Nodes().IsUserIndexed(11));
			Assert::AreEqual(true, model.Nodes().IsUserIndexed(12));

			// check that our node sets exist and has all specified nodes
			Assert::AreEqual(true, model.ExistsNodeSet(_T("nodal_load_set")));
			Assert::AreEqual(true, model.GetNodeSet(_T("nodal_load_set")).IsUserIndexed(5));
			Assert::AreEqual(true, model.GetNodeSet(_T("nodal_load_set")).IsUserIndexed(6));

			Assert::AreEqual(true, model.ExistsNodeSet(_T("all_nodes")));
			Assert::AreEqual(true, model.GetNodeSet(_T("all_nodes")).IsUserIndexed(1));
			Assert::AreEqual(true, model.GetNodeSet(_T("all_nodes")).IsUserIndexed(2));
			Assert::AreEqual(true, model.GetNodeSet(_T("all_nodes")).IsUserIndexed(3));
			Assert::AreEqual(true, model.GetNodeSet(_T("all_nodes")).IsUserIndexed(4));
			Assert::AreEqual(true, model.GetNodeSet(_T("all_nodes")).IsUserIndexed(5));
			Assert::AreEqual(true, model.GetNodeSet(_T("all_nodes")).IsUserIndexed(6));
			Assert::AreEqual(true, model.GetNodeSet(_T("all_nodes")).IsUserIndexed(7));
			Assert::AreEqual(true, model.GetNodeSet(_T("all_nodes")).IsUserIndexed(8));
			Assert::AreEqual(true, model.GetNodeSet(_T("all_nodes")).IsUserIndexed(9));
			Assert::AreEqual(true, model.GetNodeSet(_T("all_nodes")).IsUserIndexed(10));
			Assert::AreEqual(true, model.GetNodeSet(_T("all_nodes")).IsUserIndexed(11));
			Assert::AreEqual(true, model.GetNodeSet(_T("all_nodes")).IsUserIndexed(12));

			// check that every element exists
			Assert::AreEqual(true, model.Elements().IsUserIndexed(1));
			Assert::AreEqual(true, model.Elements().IsUserIndexed(2));

			// check that element sets exist and is correctly populated
			Assert::AreEqual(true, model.ExistsElementSet(_T("all_elements")));
			Assert::AreEqual(true, model.GetElementSet(_T("all_elements")).IsUserIndexed(1));
			Assert::AreEqual(true, model.GetElementSet(_T("all_elements")).IsUserIndexed(2));

			// check that the curves exist
			Assert::AreEqual(true, model.Curves().Contains(_T("constant load")));

			// check that the step exists
			Assert::AreEqual(1, analysis.GetStepCount());
			const AnalysisStep& step = analysis.GetStep(0);
			Assert::AreEqual(0, step.GetStartTime(), REAL_TOLERANCE);
			Assert::AreEqual((real)0.25, step.GetEndTime(), REAL_TOLERANCE);
			Assert::AreEqual(0, step.GetTimeline().StartTime(), REAL_TOLERANCE);
			Assert::AreEqual((real)0.25, step.GetTimeline().EndTime(), REAL_TOLERANCE);
      Assert::AreEqual(1, step.GetResults().GetChainCount());

			// check that the snapshot was added
			Assert::AreEqual(1L, step.GetTimeline().SnapshotMarkCount());
			Assert::AreEqual((real)0.25, step.GetTimeline().GetSnapshotMark(0).GetTime());

			// check that loads are applied correctly
			Node *node = &model.Nodes().GetByUserIndex(5);
			Assert::AreEqual(false, step.NodalLoads().Contains(node->GetDoF(0)));
			Assert::AreEqual(true,  step.NodalLoads().Contains(node->GetDoF(1)));
			Assert::AreEqual(false, step.NodalLoads().Contains(node->GetDoF(2)));
			node = &model.Nodes().GetByUserIndex(6);
			Assert::AreEqual(false, step.NodalLoads().Contains(node->GetDoF(0)));
			Assert::AreEqual(true,  step.NodalLoads().Contains(node->GetDoF(1)));
			Assert::AreEqual(false, step.NodalLoads().Contains(node->GetDoF(2)));

			// check that constraints were applied correctly
			node = &model.Nodes().GetByUserIndex(1);
			Assert::AreEqual(true, step.Locks().Contains(node->GetDoF(0)));
			Assert::AreEqual(true, step.Locks().Contains(node->GetDoF(1)));
			Assert::AreEqual(true, step.Locks().Contains(node->GetDoF(2)));
			node = &model.Nodes().GetByUserIndex(2);
			Assert::AreEqual(true, step.Locks().Contains(node->GetDoF(0)));
			Assert::AreEqual(true, step.Locks().Contains(node->GetDoF(1)));
			Assert::AreEqual(true, step.Locks().Contains(node->GetDoF(2)));
			node = &model.Nodes().GetByUserIndex(7);
			Assert::AreEqual(true, step.Locks().Contains(node->GetDoF(0)));
			Assert::AreEqual(true, step.Locks().Contains(node->GetDoF(1)));
			Assert::AreEqual(true, step.Locks().Contains(node->GetDoF(2)));
			node = &model.Nodes().GetByUserIndex(8);
			Assert::AreEqual(true, step.Locks().Contains(node->GetDoF(0)));
			Assert::AreEqual(true, step.Locks().Contains(node->GetDoF(1)));
			Assert::AreEqual(true, step.Locks().Contains(node->GetDoF(2)));

			// other nodes must be free (of boundary conditions)
			for (int i = 3; i <= 12; i++)
			{
				if (i != 5 && i != 6 && i != 7 && i != 8)
				{
					node = &model.Nodes().GetByUserIndex(i);
					Assert::AreEqual(false, step.DefinesBoundaryCondition(node->GetDoF(0)));
					Assert::AreEqual(false, step.DefinesBoundaryCondition(node->GetDoF(1)));
					Assert::AreEqual(false, step.DefinesBoundaryCondition(node->GetDoF(2)));
				}
			}

			app.Destroy();
			appLog.Destroy();
			analysisLog.Destroy();
		}
	};

} } }

#endif
