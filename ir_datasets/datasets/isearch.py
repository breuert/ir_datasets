import io
import tarfile
from typing import NamedTuple
import ir_datasets
from ir_datasets.indices import PickleLz4FullStore
from ir_datasets.util import Lazy, DownloadConfig, Migrator
from ir_datasets.datasets.base import Dataset, FilteredQueries, FilteredQrels, YamlDocumentation
from ir_datasets.formats import BaseDocs, BaseQueries, BaseQrels, GenericQuery, GenericQrel, TrecQueries, TrecQrels, TrecXmlQueries


# A unique identifier for this dataset. This should match the file name (with "-" instead of "_")
NAME = 'isearch'

# What do the relevance levels in qrels mean?
QREL_DEFS = {
    2: 'highly relevant',
    1: 'relevant',
    0: 'not relevant',
}

QTYPE_MAP = {
    '<num>': 'query_id',
    '<infoneed>': 'infoneed',
    '<task>': 'task',
    '<background>': 'background',
    '<ideal>': 'ideal',
    '<keywords>': 'keywords'
}

# This message is shown to the user before downloads are started
DUA = 'Please confirm that you agree to the data usage agreement at https://sites.google.com/view/isearch-testcollection/'


class iSearchQuery(NamedTuple):
    query_id: str
    infoneed: str
    keywords: str
    task: str
    background: str
    ideal: str
    def default_text(self):
        """
        title
        """
        return self.keywords


class iSearchDoc(NamedTuple):
    doc_id: str
    document_link: str
    category: str
    title: str
    author: str
    subject: str
    description: str
    venue: str
    fulltext: str
    document_type: str
    def default_text(self):
        """
        title + description + document_link
        """
        return f'{self.title} {self.document_link}'


class iSearchDocs(BaseDocs):
    def __init__(self, dlc):
        self._dlc = dlc

    def docs_path(self, force=True):
        return self._dlc.path(force)

    def docs_cls(self):
        return iSearchDoc

    def docs_iter(self):
        return iter(self.docs_store())

    def _docs_iter(self):
        BeautifulSoup = ir_datasets.lazy_libs.bs4().BeautifulSoup
        with self._dlc.stream() as stream:
            with tarfile.open(fileobj=stream, mode='r|gz') as tgz_outer:
                for member_o in tgz_outer:
                    if not member_o.isfile() or not (member_o.name.endswith('.tar') or member_o.name.endswith('.tgz')):
                        continue
                    file = tgz_outer.extractfile(member_o)
                    with tarfile.open(fileobj=file, mode='r|gz' if member_o.name.endswith('.tgz') else 'r|') as tgz_inner:
                        for member_i in tgz_inner:
                            if not member_i.isfile():
                                continue
                            full_xml = tgz_inner.extractfile(member_i).read()
                            soup = BeautifulSoup(full_xml, 'lxml-xml')
                            doc_id = soup.find('DOCNO')
                            doc_id = doc_id.text.strip() if doc_id else ''
                            document_link = soup.find('DOCUMENTLINK')
                            document_link = document_link.text.strip() if document_link else ''
                            category = soup.find('CATEGORY')
                            category = category.text.strip() if category else ''
                            title = soup.find('TITLE')
                            title = title.text.strip() if title else ''
                            author = soup.find('AUTHOR')
                            author = author.text.strip() if author else ''
                            subject = soup.find('SUBJECT')
                            subject = subject.text.strip() if subject else ''
                            description = soup.find('DESCRIPTION')
                            description = description.text.strip() if description else ''
                            venue = soup.find('VENUE')
                            venue = venue.text.strip() if venue else ''
                            fulltext = soup.find('FULLTEXT')
                            fulltext = fulltext.text.strip() if fulltext else ''
                            document_type = soup.find('TYPE')
                            document_type = document_type.text.strip() if document_type else ''

                            yield iSearchDoc(doc_id, document_link, category, title, author, subject, description, venue, fulltext, document_type)


    def docs_store(self, field='doc_id'):
        return PickleLz4FullStore(
            path=f'{self.docs_path()}.pklz4',
            init_iter_fn=self._docs_iter,
            data_cls=self.docs_cls(),
            lookup_field=field,
            index_fields=['doc_id'],
            count_hint=ir_datasets.util.count_hint(NAME),
        )

    def docs_count(self):
        if self.docs_store().built():
            return self.docs_store().count()

    def docs_namespace(self):
        return NAME

    def docs_lang(self):
        return 'en'


# An initialization function is used to keep the namespace clean
def _init():
    # The directory where this dataset's data files will be stored
    base_path = ir_datasets.util.home_path() / NAME
    
    # Load an object that is used for providing the documentation
    documentation = YamlDocumentation(f'docs/{NAME}.yaml')
    
    # A reference to the downloads file, under the key "dummy". (DLC stands for DownLoadable Content)
    dlc = DownloadConfig.context(NAME, base_path, dua=DUA)
    
    # How to process the documents. Since they are in a typical TSV format, we'll use TsvDocs.
    # Note that other dataset formats may require you to write a custom docs handler (BaseDocs).
    # Note that this doesn't process the documents now; it just defines how they are processed.
    docs = iSearchDocs(dlc['docs'])
    
    # How to process the queries. Similar to the documents, you may need to write a custom
    # queries handler (BaseQueries).
    queries = TrecQueries(dlc['queries'], namespace=NAME, qtype_map=QTYPE_MAP, qtype=iSearchQuery, lang='en')

    # Qrels: The qrels file is in the TREC format, so we'll use TrecQrels to process them
    qrels = TrecQrels(dlc['qrels'], QREL_DEFS)

    # Package the docs, queries, qrels, and documentation into a Dataset object
    dataset = Dataset(docs, queries, qrels, documentation('_'))
    
    # Register the dataset in ir_datasets
    ir_datasets.registry.register(NAME, dataset)
    
    return dataset # used for exposing dataset to the namespace

dataset = _init()