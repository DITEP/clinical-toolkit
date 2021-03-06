"""
This script is to test text_parser.text_parser script

@TODO tests for unstructured html
"""
from bs4 import BeautifulSoup
from nose.tools import assert_equal
from clintk.text_parser import parser_utils

html_doc = """
<h3> Some title 0</h3>  
<span style= 'color: red'> texts in red are important   </span> 
<div> This is div 0</div> 
<p> text 0 0 </p> 
<p> text 0 1 </p> 
<h3> title 1</h3>
<div> This is div 1</div>
<p> text 1 0 </p>
<p> text 1 1 </p>
<span> this is a span</span>

"""


class TestParser(object):
    def setUp(self):
        """ reinstanciate the soup before each
        test

        Returns
        -------

        """
        soup = BeautifulSoup(html_doc, 'html.parser')
        return soup, list(soup.find_all('h3'))

    def test_empty_cleaner(self):
        soup, _ = self.setUp()
        parser_utils.clean_soup(soup, remove=['div'], verbose=False)

        expected = """
        <h3> Some title 0</h3>
        
        
        <p> text 0 0 </p>
        <p> text 0 1 </p>
        <h3> title 1</h3>
        
        <p> text 1 0 </p>
        <p> text 1 1 </p>
        <span> this is a span</span>
        """
        assert_equal(soup.prettify(),
                     BeautifulSoup(expected, 'html.parser').prettify())

    def test_clean_string(self):
        s1 = 'Hello World'
        s2 = 'Hello World! 123      '
        s3 = 'Bonjour à tous'

        s1_clean = parser_utils.clean_string(s1)
        s2_clean = parser_utils.clean_string(s2)
        s3_clean = parser_utils.clean_string(s3)

        assert_equal('hello world', s1_clean)
        assert_equal('hello world 123', s2_clean)
        assert_equal('bonjour a tous', s3_clean)

    def test_parse(self):
        soup, _ = self.setUp()

        res = parser_utils.parse_soup(soup, True, False, 'h3')
        expected = {'some title 0': 'texts in red are important this ' +
                                    'is div 0 text 0 0 text 0 1',
                    'title 1': 'this is div 1 text 1 0 text 1 1 ' +
                               'this is a span'}
        assert_equal(res, expected)

    def test_last_tag(self):
        soup, tags = self.setUp()
        last_tag = tags[-1]

        res_text = parser_utils.last_tag_text(last_tag, True)
        expected = 'this is div 1 text 1 0 text 1 1 this is a span'

        assert_equal(res_text, expected)

    def test_between_tags(self):
        soup, tags = self.setUp()

        res = parser_utils.text_between_tags(tags[0].find_next(), tags[1],
                                             True)
        expected = 'texts in red are important this is div 0 ' \
                   'text 0 0 text 0 1'

        assert_equal(res, expected)

    def test_main_parser(self):
        # soup, _ = self.setUp()
        res = parser_utils.main_parser(html_doc, True, False, [], 'h3')

        expected = {'some title 0': 'this is div 0 text ' +
                                    '0 0 text 0 1',
                    'title 1': 'this is div 1 text 1 0 text 1 1 ' +
                               'this is a span'}

        assert_equal(res, expected)




















