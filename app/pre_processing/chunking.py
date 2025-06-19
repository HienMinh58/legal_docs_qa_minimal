from semantic_text_splitter import TextSplitter
from langchain.text_splitter import NLTKTextSplitter

class Chunking:
    '''Initializes the class with specified maximum characters and overlap size for text splitting.

    Args:
        max_characters (int, optional): The maximum number of characters for each chunk. Defaults to 1000.
        overlap_size (int, optional): The number of characters to overlap between chunks. Defaults to 50.
    '''
    def __init__(self, max_characters=1000, overlap_size=50):
        self.chunk_splitter = TextSplitter(max_characters)
        self.overlap_splitter = TextSplitter(overlap_size)
    
    def split_document_with_order_overlap(self, document_text):
        """Splits a document into chunks with overlapping content between consecutive chunks.

        This method uses two splitters: one for the main chunking and another for creating the overlap.
        Each chunk, except the first one, will have an overlap from the end of the previous chunk.

        Args:
            document_text (str): The text of the document to be split.

        Returns:
            list of dict: A list of dictionaries, each containing a chunk of the document and its corresponding chunk ID.
            Each dictionary has the following keys:
                - 'content' (str): The text content of the chunk.
                - 'chunk_id' (int): The ID of the chunk, starting from 1.
        """
        chunks = self.chunk_splitter.chunks(document_text)
        
        result = []
        for idx, chunk in enumerate(chunks):
            if idx > 0:
                overlap = self.overlap_splitter.chunks(chunks[idx-1])[-1]
                chunk = overlap + chunk
            
            result.append({'content': chunk, 'chunk_id': idx + 1})
        
        return result


    def split_document_with_title_overlap(self, document_text, title):
        """
        Splits a document into chunks with title overlap.

        This method splits the given document text into smaller chunks using a specified separator.
        It ensures that important sections, such as '- Liều dùng: ...', are not split inappropriately.
        Each chunk is overlapped with the provided title to maintain context.

        Args:
            document_text (str): The text of the document to be split.
            title (str): The title to be used as overlap for each chunk.

        Returns:
            list of dict: A list of dictionaries, each containing a chunk of the document and its corresponding chunk ID.
            Each dictionary has the following keys:
                - 'content' (str): The text content of the chunk.
                - 'chunk_id' (int): The ID of the chunk, starting from 1.
        """
        # Tách document thành các chunk bằng cách sử dụng dấu '-' làm điểm tách
        # Tránh tách các phần quan trọng như 'Liều dùng: ...' bị cắt
        # Split document by the separator '-'
        chunks = self.chunk_splitter.chunks(document_text)

        result = []
        overlap_chunk = title  # Sử dụng title webname làm overlap

        for idx, chunk in enumerate(chunks):
            # Khi không phải chunk đầu tiên, chúng ta thêm overlap
            if idx > 0:
                chunk = overlap_chunk + chunk
            
            result.append({'content': chunk, 'chunk_id': idx + 1})

            # Update overlap_chunk để sử dụng webname cho chunk tiếp theo
            overlap_chunk = title  # Điều chỉnh overlap luôn là title webname trong tất cả các chunk

        return result
    
    def chunk_ppt(self, document_text):
        text_splitter = NLTKTextSplitter(chunk_size=1000)
        chunks = text_splitter.split_text(document_text)
            
        result = []
        for idx, chunk in enumerate(chunks):
            result.append({'content': chunk, 'chunk_id': idx + 1})
        
        return result
    
    def split_by_separator(self, text, separator='***'):
        """Split text by specified separator and return chunks with content and chunk_id.
        
        Args:
            text: The text to split
            separator: The separator to split the text by (default: newline)
            
        Returns:
            List of dictionaries containing 'content' and 'chunk_id' for each chunk
        """
        chunks = [chunk.strip() for chunk in text.split(separator) if chunk.strip()]
        return [{'content': chunk, 'chunk_id': idx + 1} for idx, chunk in enumerate(chunks)]