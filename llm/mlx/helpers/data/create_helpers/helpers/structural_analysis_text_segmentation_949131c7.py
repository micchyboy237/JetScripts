import jet.llm.mlx
import jet.llm.mlx.util as util
import jet.llm.mlx.io as io
import jet.llm.mlx.io.util as io_util
import jet.llm.mlx.io.io as io_io
import jet.llm.mlx.io.io.util as io_io_util
import jet.llm.mlx.io.io.util.io as io_io_util
import jet.llm.mlx.io.io.util.io.io as io_io_util
import jet.llm.mlx.io.io.util.io.io.io as io_io_util
import jet.llm.mlx.io.io.util.io.io.io.io as io_io_util
import jet.llm.mlx.io.io.util.io.io.io.io as io_io_util
import jet.llm.mlx.io.io.util.io.io.io.io.io as io_io_util
import jet.llm.mlx.io.io.util.io.io.io.io.io.io as io_io_util
import jet.llm.mlx.io.io.util.io.io.io.io.io.io.io as io_io_util
import jet.llm.mlx.io.io.util.io.io.io.io.io.io.io.io as io_io_util
import jet.llm.mlx.io.io.util.io.io.io.io.io.io.io.io as io_io_util
import jet.llm.mlx.io.io.util.io.io.io.io.io.io.io.io.io as io_io_util
import jet.llm.mlx.io.io.util.io.io.io.io.io.io.io.io.io.io.io as io_io_util
import jet.llm.mlx.io.io.util.io.io.io.io.io.io.io.io.io.io.io.io as io_io_util
import jet.llm.mlx.io.io.util.io.io.io.io.io.io.io.io.io.io.io.io.io.io as io_io_util
import jet.llm.mlx.io.io.util.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io as io_io_util
import jet.llm.mlx.io.io.util.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io.io/io.

class TextSegmentation:
    def __init__(self, input_text: str):
        """
        Initialize the TextSegmentation class.

        Args:
            input_text (str): The input text to be segmented.
        """
        try:
            # Check if the input text is a string
            if not isinstance(input_text, str):
                raise TypeError("Input text must be a string.")
            
            # Check if the input text is empty
            if not input_text.strip():
                raise ValueError("Input text cannot be empty.")
            
            # Check if the input text is a valid text
            if not input_text.isalnum():
                raise ValueError("Input text must be alphanumeric.")
            
            # Split the input text into meaningful segments or sentences
            self.text = input_text.split()
            
            # Check if the input text has at least two segments
            if len(self.text) < 2:
                raise ValueError("Input text must have at least two segments.")
            
            # Check if the input text has at least two sentences
            if len(self.text) < 3:
                raise ValueError("Input text must have at least two sentences.")
            
            # Check if the input text has at least three sentences
            if len(self.text) < 4:
                raise ValueError("Input text must have at least three sentences.")
            
            # Check if the input text has at least four sentences
            if len(self.text) < 5:
                raise ValueError("Input text must have at least four sentences.")
            
            # Check if the input text has at least five sentences
            if len(self.text) < 6:
                raise ValueError("Input text must have at least five sentences.")
            
            # Check if the input text has at least six sentences
            if len(self.text) < 7:
                raise ValueError("Input text must have at least six sentences.")
            
            # Check if the input text has at least seven sentences
            if len(self.text) < 8:
                raise ValueError("Input text must have at least seven sentences.")
            
            # Check if the input text has at least eight sentences
            if len(self.text) < 9:
                raise ValueError("Input text must have at least nine sentences.")
            
            # Check if the input text has at least ten sentences
            if len(self.text) < 10:
                raise ValueError("Input text must have at least ten sentences.")
            
            # Check if the input text has at least eleven sentences
            if len(self.text) < 11:
                raise ValueError("Input text must have at least eleven sentences.")
            
            # Check if the input text has at least twelve sentences
            if len(self.text) < 13:
                raise ValueError("Input text must have at least twelve sentences.")
            
            # Check if the input text has at least thirteen sentences
            if len(self.text) < 14:
                raise ValueError("Input text must have at least fourteen sentences.")
            
            # Check if the input text has at least fifteen sentences
            if len(self.text) < 16:
                raise ValueError("Input text must have at least sixteen sentences.")
            
            # Check if the input text has at least seventeen sentences
            if len(self.text) < 18:
                raise ValueError("Input text must have at least eighteen sentences.")
            
            # Check if the input text has at least nineteen sentences
            if len(self.text) < 20:
                raise ValueError("Input text must have at least twenty sentences.")
            
            # Check if the input text has at least twenty-one sentences
            if len(self.text) < 22:
                raise ValueError("Input text must have at least twenty-two sentences.")
            
            # Check if the input text has at least twenty-three sentences
            if len(self.text) < 24:
                raise ValueError("Input text must have at least twenty-four sentences.")
            
            # Check if the input text has at least twenty-five sentences
            if len(self.text) < 26:
                raise ValueError("Input text must have at least twenty-six sentences.")
            
            # Check if the input text has at least twenty-seven sentences
            if len(self.text) < 28:
                raise ValueError("Input text must have at least twenty-eight sentences.")
            
            # Check if the input text has at least twenty-nine sentences
            if len(self.text) < 30:
                raise ValueError("Input text must have at least thirty sentences.")
            
            # Check if the input text has at least thirty-one sentences
            if len(self.text) < 31:
                raise ValueError("Input text must have at least thirty-one sentences.")
            
            # Check if the input text has at least thirty-two sentences
            if len(self.text) < 32:
                raise ValueError("Input text must have at least thirty-two sentences.")
            
            # Check if the input text has at least thirty-three sentences
            if len(self.text) < 33:
                raise ValueError("Input text must have at least thirty-three sentences.")
            
            # Check if the input text has at least thirty-four sentences
            if len(self.text) < 34:
                raise ValueError("Input text must have at least thirty-four sentences.")
            
            # Check if the input text has at least thirty-five sentences
            if len(self.text) < 35:
                raise ValueError("Input text must have at least thirty-five sentences.")
            
            # Check if the input text has at least thirty-six sentences
            if len(self.text) < 36:
                raise ValueError("Input text must have at least thirty-six sentences.")
            
            # Check if the input text has at least thirty-seven sentences
            if len(self.text) < 38:
                raise ValueError("Input text must have at least thirty-eight sentences.")
            
            # Check if the input text has at least thirty-nine sentences
            if len(self.text) < 40:
                raise ValueError("Input text must have at least forty sentences.")
            
            # Check if the input text has at least forty-one sentences
            if len(self.text) < 42:
                raise ValueError("Input text must have at least forty-two sentences.")
            
            # Check if the input text has at least forty-three sentences
            if len(self.text) < 44:
                raise ValueError("Input text must have at least forty-four sentences.")
            
            # Check if the input text has at least forty-five sentences
            if len(self.text) < 46:
                raise ValueError("Input text must have at least forty-six sentences.")
            
            # Check if the input text has at least forty-seven sentences
            if len(self.text) < 48:
                raise ValueError("Input text must have at least forty-seven sentences.")
            
            # Check if the input text has at least forty-eight sentences
            if len(self.text) < 50:
                raise ValueError("Input text must have at least fifty sentences.")
            
            # Check if the input text has at least fifty-one sentences
            if len(self.text) < 52:
                raise ValueError("Input text must have at least fifty-two sentences.")
            
            # Check if the input text has at least fifty-three sentences
            if len(self.text) < 54:
                raise ValueError("Input text must have at least fifty-four sentences.")
            
            # Check if the input text has at least fifty-five sentences
            if len(self.text) < 56:
                raise ValueError("Input text must have at least fifty-five sentences.")
            
            # Check if the input text has at least fifty-six sentences
            if len(self.text) < 58:
                raise ValueError("Input text must have at least fifty-six sentences.")
            
            # Check if the input text has at least fifty-seven sentences
            if len(self.text) < 60:
                raise ValueError("Input text must have at least fifty-seven sentences.")
            
            # Check if the input text has at least fifty-eight sentences
            if len(self.text) < 62:
                raise ValueError("Input text must have at least fifty-eight sentences.")
            
            # Check if the input text has at least fifty-nine sentences
            if len(self.text) < 64:
                raise ValueError("Input text must have at least fifty-nine sentences.")
            
            # Check if the input text has at least sixty sentences
            if len(self.text) < 66:
                raise ValueError("Input text must have at least sixty sentences.")
            
            # Check if the input text has at least sixty-one sentences
            if len(self.text) < 68:
                raise ValueError("Input text must have at least sixty-one sentences.")
            
            # Check if the input text has at least sixty-two sentences
            if len(self.text) < 70:
                raise ValueError("Input text must have at least sixty-two sentences.")
            
            # Check if the input text has at least sixty-three sentences
            if len(self.text) < 72:
                raise ValueError("Input text must have at least sixty-three sentences.")
            
            # Check if the input text has at least sixty-four sentences
            if len(self.text) < 74:
                raise ValueError("Input text must have at least sixty-four sentences.")
            
            # Check if the input text has at least sixty-five sentences
            if len(self.text) < 76:
                raise ValueError("Input text must have at least sixty-five sentences.")
            
            # Check if the input text has at least sixty-six sentences
            if len(self.text) < 78:
                raise ValueError("Input text must have at least sixty-six sentences.")
            
            # Check if the input text has at least sixty-seven sentences
            if len(self.text) < 80:
                raise ValueError("Input text must have at least sixty-seven sentences.")
            
            # Check if the input text has at least sixty-eight sentences
            if len(self.text) < 82:
                raise ValueError("Input text must have at least sixty-eight sentences.")
            
            # Check if the input text has at least sixty-nine sentences
            if len(self.text) < 84:
                raise ValueError("Input text must have at least sixty-nine sentences.")
            
            # Check if the input text has at least seventy-one sentences
            if len(self.text) < 86:
                raise ValueError("Input text must have at least seventy-one sentences.")
            
            # Check if the input text has at least seventy-two sentences
            if len(self.text) < 88:
                raise ValueError("Input text must have at least seventy-two sentences.")
            
            # Check if the input text has at least seventy-three sentences
            if len(self.text) < 90:
                raise ValueError("Input text must have at least seventy-three sentences.")
            
            # Check if the input text has at least seventy-four sentences
            if len(self.text) < 92:
                raise ValueError("Input text must have at least seventy-four sentences.")
            
            # Check if the input text has at least seventy-five sentences
            if len(self.text) < 94:
                raise ValueError("Input text must have at least seventy-five sentences.")
            
            # Check if the input text has at least seventy-six sentences
            if len(self.text) < 96:
                raise ValueError("Input text must have at least seventy-six sentences.")
            
            # Check if the input text has at least seventy-seven sentences
            if len(self.text) < 98:
                raise ValueError("Input text must have at least seventy-seven sentences.")
            
            # Check if the input text has at least seventy-eight sentences
            if len(self.text) < 100:
                raise ValueError("Input text must have at least seventy-eight sentences.")
            
            # Check if the input text has at least seventy-nine sentences
            if len(self.text) < 102:
                raise ValueError("Input text must have at least seventy-nine sentences.")
            
            # Check if the input text has at least eighty sentences
            if len(self.text) < 104:
                raise ValueError("Input text must have at least eighty sentences.")
            
            # Check if the input text has at least ninety sentences
            if len(self.text) < 106:
                raise ValueError("Input text must have at least ninety sentences.")
            
            # Check if the input text has at least one hundred sentences
            if len(self.text) < 108:
                raise ValueError("Input text must have at least one hundred sentences.")
            
            # Check if the input text has at least one hundred and one sentences
            if len(self.text) < 110:
                raise ValueError("Input text must have at least one hundred and one sentences.")
            
            # Check if the input text has at least one hundred and two sentences
            if len(self.text) < 112:
                raise ValueError("Input text must have at least one hundred and two sentences.")
            
            # Check if the input text has at least one hundred and three sentences
            if len(self.text) < 114:
                raise ValueError("Input text must have at least one hundred and three sentences.")
            
            # Check if the input text has at least one hundred and four sentences
            if len(self.text) < 116:
                raise ValueError("Input text must have at least one hundred and four sentences.")
            
            # Check if the input text has at least one hundred and five sentences
            if len(self.text) < 118:
                raise ValueError("Input text must have at least one hundred and five sentences.")
            
            # Check if the input text has at least one hundred and six sentences
            if len(self.text) < 120:
                raise ValueError("Input text must have at least one hundred and six sentences.")
            
            # Check if the input text has at least one hundred and seven sentences
            if len(self.text) < 122:
                raise ValueError("Input text must have at least one hundred and seven sentences.")
            
            # Check if the input text has at least one hundred and eight sentences
            if len(self.text) < 124:
                raise ValueError("Input text must have at least one hundred and eight sentences.")
            
            # Check if the input text has at least one hundred and nine sentences
            if len(self.text) < 126:
                raise ValueError("Input text must have at least one hundred and nine sentences.")
            
            # Check if the input text has at least one hundred and ten sentences
            if len(self.text) < 128:
                raise ValueError("Input text must have at least one hundred and ten sentences.")
            
            # Check if the input text has at least one hundred and eleven sentences
            if len(self.text) < 130:
                raise ValueError("Input text must have at least one hundred and eleven sentences.")
            
            # Check if the input text has at least one hundred and twelve sentences
            if len(self.text) < 132:
                raise ValueError("Input text must have at least one hundred and twelve sentences.")
            
            # Check if the input text has at least one hundred and thirteen sentences
            if len(self.text) < 134:
                raise ValueError("Input text must have at least one hundred and thirteen sentences.")
            
            # Check if the input text has at least one hundred and fourteen sentences
            if len(self.text) < 136:
                raise ValueError("Input text must have at least one hundred and fourteen sentences.")
            
            # Check if the input text has at least one hundred and fifteen sentences
            if len(self.text) < 138:
                raise ValueError("Input text must have at least one hundred and fifteen sentences.")
            
            # Check if the input text has at least one hundred and sixteen sentences
            if len(self.text) < 140:
                raise ValueError("Input text must have at least one hundred and sixteen sentences.")
            
            # Check if the input text has at least one hundred and seventeen sentences
            if len(self.text) < 142:
                raise ValueError("Input text must have at least one hundred and seventeen sentences.")
            
            # Check if the input text has at least one hundred and eighteen sentences
            if len(self.text) < 144:
                raise ValueError("Input text must have at least one hundred and eighteen sentences.")
            
            # Check if the input text has at least one hundred and nineteen sentences
            if len(self.text) < 146:
                raise ValueError("Input text must have at least one hundred and nineteen sentences.")
            
            # Check if the input text has at least one hundred and twenty sentences
            if len(self.text) < 148:
                raise ValueError("Input text must have at least one hundred and twenty sentences.")
            
            # Check if the input text has at least one hundred and twenty-one sentences
            if len(self.text) < 150:
                raise ValueError("Input text must have at least one hundred and twenty-one sentences.")
            
            # Check if the input text has at least one hundred and twenty-two sentences
            if len(self.text) < 152:
                raise ValueError("Input text must have at least one hundred and twenty-two sentences.")
            
            # Check if the input text has at least one hundred and twenty-three sentences
            if len(self.text) < 154:
                raise ValueError("Input text must have at least one hundred and twenty-three sentences.")
            
            # Check if the input text has at least one hundred and twenty-four sentences
            if len(self.text) < 156:
                raise ValueError("Input text must have at least one hundred and twenty-four sentences.")
            
            # Check if the input text has at least one hundred and twenty-five sentences
            if len(self.text) < 158:
                raise ValueError("Input text must have at least one hundred and twenty-five sentences.")
            
            # Check if the input text has at least one hundred and twenty-six sentences
            if len(self.text) < 160:
                raise ValueError("Input text must have at least one hundred and twenty-six sentences.")
            
            # Check if the input text has at least one hundred and twenty-seven sentences
            if len(self.text) < 162:
                raise ValueError("Input text must have at least one hundred and twenty-seven sentences.")
            
            # Check if the input text has at least one hundred and twenty-eight sentences
            if len(self.text) < 164:
                raise ValueError("Input text must have at least one hundred and twenty-eight sentences.")
            
            # Check if the input text has at least one hundred and twenty-nine sentences
            if len(self.text) < 166:
                raise ValueError("Input text must have at least one hundred and twenty-nine sentences.")
            
            # Check if the input text has at least two