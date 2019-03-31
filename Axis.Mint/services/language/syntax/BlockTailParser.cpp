#include "BlockTailParser.hpp"
#include "../factories/AxisGrammar.hpp"
#include "../grammar_tokens.hpp"
#include "services/language/parsing/ExpressionNode.hpp"

namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace asls = axis::services::language::syntax;

asls::BlockTailParser::BlockTailParser( void )
{
	// init grammar
	_parser << aslf::AxisGrammar::CreateBlankParser()
		      << aslf::AxisGrammar::CreateReservedWordBlockTailParser()
			    << aslf::AxisGrammar::CreateBlankParser(true)
			    << aslf::AxisGrammar::CreateIdParser() 
			    << aslf::AxisGrammar::CreateBlankParser();
}

asls::BlockTailParser::~BlockTailParser( void )
{
	// nothing to do
}

axis::String asls::BlockTailParser::GetBlockName( void ) const
{
	return _blockName;
}

aslp::ParseResult asls::BlockTailParser::Parse( const asli::InputIterator& begin, 
                                                const asli::InputIterator& end )
{
	aslp::ParseResult result = _parser(begin, end, false);
	if (result.IsMatch())
	{	// get id node
    aslp::ExpressionNode& rootNode = (aslp::ExpressionNode&)result.GetParseTree();
    aslp::ParseTreeNode *idNode = rootNode.GetFirstChild()->GetNextSibling();
		_blockName = idNode->ToString();
	}
	else
	{
		_blockName = _T("");
	}
	return result;
}

aslp::ParseResult asls::BlockTailParser::operator()( const asli::InputIterator& begin, 
                                                     const asli::InputIterator& end )
{
	return Parse(begin, end);
}
