"""
This script contains the functions used to parse one report, ie the functions
to split the html text into a dictionnary of sections.

Only main_parser is used in practice since all the other functions
are auxiliary. Moreover, they should not be used "as-is" since they are
wrapped up in the `ReportsParser` object for convenience.

@TODO change module name to `utils`
"""
from bs4 import BeautifulSoup

from unidecode import unidecode
import re


def main_parser(text, is_html, verbose, remove, headers):
    """ takes as input the string from the report and splits it into sections

    Parameters
    ----------
    text : string
        report in html format

    is_html : bool
        set True if text is actually structured as html

    verbose : bool
        True for logging

    remove : list
        name of the tags to remove because contain useless information

    headers : string
        name of the tags that delimit the sections

    Returns
    -------
    dict
        keys are section names, values are the content of the section

    """
    try:
        soup = BeautifulSoup(text, 'html.parser')
        soup = BeautifulSoup(soup.prettify(), 'html.parser')
    except TypeError:
        if verbose:
            print('{} can not be parsed'.format(text))
        soup = BeautifulSoup('', 'html.parser')

    clean_soup(soup, remove, verbose)

    return parse_soup(soup, is_html, verbose, headers)


def text_between_tags(tag1, tag2, is_html):
    """ This function fetches the text between tag 1 and tag 2

    The soup should already be cleansed from useless tags such as  span

    Parameters
    ----------
    tag1

    tag2

    is_html

    Returns
    -------
    str
        all the text between tag1 and tag2

    """
    if is_html:
        res = tag1.text
        next_tag = tag1.find_next()
        # iterates over tags to append text to res
        while next_tag != tag2:
            res += next_tag.text + ' '
            next_tag = next_tag.find_next()

        return clean_string(res)

    else:
        res = tag1.next_sibling.strip()
        next_tag = tag1.find_next()
        while next_tag != tag2:
            res += next_tag.next_sibling.strip() + ' '
            next_tag = next_tag.find_next()

        return clean_string(res)


def last_tag_text(final_tag, is_html):
    """ Fetches text from last tag

    Parameters
    ----------
    final_tag

    Returns
    -------
    string
        content of the last section

    """
    if is_html:
        res = ''
        cur_tag = final_tag.find_next()
        while cur_tag is not None:
            res += cur_tag.text + ' '
            cur_tag = cur_tag.find_next()
        return clean_string(res)
    else:
        return clean_string(final_tag.next_sibling)


def parse_soup(soup, is_html, verbose, headers='h3'):
    """Splits the soup between headers and returns a dictionnary

    Parameters
    ----------
    soup : BeautifulSoup

    is_html : bool
        true if text is exact html format

    verbose: bool, (default=False)
        weather to print information about parsing

    headers : string
        delimiters of the sections

    Returns
    -------
    dict
        keys are section names values are section contents

    """

    res_dic = {}
    header_list = list(soup.find_all(headers))

    for index, header in enumerate(header_list[:-1]):
        try:
            if is_html:
                new_text = text_between_tags(header.find_next(),
                                             header_list[index + 1],
                                             is_html)
            else:
                new_text = text_between_tags(header,
                                             header_list[index + 1],
                                             is_html)
            key = header.text
            res_dic[clean_string(key)] = new_text
        except AttributeError as e:
            # @TODO fix verbosity
            print('{} occurred at {}'.format(e, soup.name))
    try:
        final_text = last_tag_text(header_list[-1], is_html)
        final_key = header_list[-1].text
        res_dic[clean_string(final_key)] = final_text
    except IndexError as e:
        if verbose:
            print('{} current report {} is empty'.format(e, soup.name))
    if verbose:
        print('Document {} has been parsed'.format(soup.name))
    return res_dic


def clean_soup(soup, remove, verbose):
    """ Remove the tags indicated in remove parameter from the soup
    @TODO change function name to `to_alpha_num`
    Transfo done inplace

    Parameters
    ----------
    soup : BeautifulSoup instance

    remove : list
        name of the tags to remove from the soup

    verbose: bool
        controls logging

    Returns
    -------
    BeautifulSoup
        the same as input, transformation is done inplace

    """
    # remove first span <span style= "color: red"> that indicates
    # color legend
    try:
        soup.find('span', attrs={'style': "color: red"}).extract()
    except AttributeError:
        if verbose:
            print('No legend found')

    # remove tags indicated in input
    for tag in remove:
        cont = True
        while cont:
            try:
                soup.find(tag).extract()
            except AttributeError:
                if verbose:
                    print('No tag {} in the soup {}'.format(tag, soup.name))
                cont = False
    return


def clean_string(s):
    """ remove non alphanumeric characters from string s
    returns the lowerCase string

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        string with only alphanumeric and lowercased
    """
    try:
        s_decoded = unidecode(s).replace('\n', '').replace('  ', ' ')
        pattern = re.compile('[\W_]+')
        return pattern.sub(' ', s_decoded).lower().strip()
    except:
        return s or ''
