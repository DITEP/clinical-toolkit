"""
Script to test HTMLParser object
"""
import pandas as pd

from bs4 import BeautifulSoup
from nose.tools import assert_equal, assert_list_equal
from reptk.features.transformers import HTMLParser


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

class TestHTMLParser(object):
    def setUp(self):
        """ reinstanciate the soup before each test

        Returns
        -------
        BeautifulSoup
        list
            list of h3 tags
        """
        return pd.Series([html_doc])


    def test_transform(self):
        x = self.setUp()

        parsed_x = HTMLParser(strategy='strings').transform(x)
        res_text = parsed_x.values[0]

        expected_text = 'some title 0 this is div 0 text 0 0 text 0 1 title' \
                        ' 1 this is div 1 text 1 0 text 1 1 this is a span'

        assert_equal(res_text, expected_text)


    def test_tranform_tokens(self):
        x = self.setUp()
        parsed_x = HTMLParser(strategy='tokens').transform(x)

        res_tokens = parsed_x.values[0]

        expected_tokens = ['some', 'title', '0', 'this', 'is', 'div', '0',
                           'text', '0', '0', 'text', '0', '1', 'title',
                           '1', 'this', 'is', 'div', '1', 'text', '1', '0',
                           'text', '1', '1', 'this', 'is', 'a', 'span']

        assert_list_equal(res_tokens, expected_tokens)









