import csv
from pathlib import Path
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
import re




class CSVLogger:
        
    @classmethod
    def validate_str(cls, value):
        """
        Rejects strings with: , " \n \r \t ; \
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected a string, got {type(value).__name__}")

        if re.search(r'[,"\n\r\t;\\]', value):
            raise ValueError(f"Invalid string for CSV: contains forbidden characters: {value}")

        return value
    
    @classmethod
    def validate(cls, value):
        """
        Validates if a value can be safely written to a CSV.
        Accepts: str (no ,"\n\r\t;\\), int, float, bool, None.
        Returns value or raises ValueError.
        """
        if value is None:
            return value

        if isinstance(value, (int, float, bool)):
            return value

        if isinstance(value, str):
            return cls.validate_str(value)

        raise ValueError(f"Unsupported type for CSV field: {type(value).__name__}")
    
    @classmethod
    def validate_header(cls, header):
        if not isinstance(header, list) or not header:
            raise ValueError("Header must be a non-empty list.")
        return [cls.validate_str(item) for item in header]
    
    
    def __init__(self, filepath, header=None):
        self.filepath = Path(filepath).resolve()
        if not self.filepath.parent.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.filepath.parent}")

        if self.filepath.exists():
            # Read and validate existing header
            with self.filepath.open(mode='r', encoding="utf-8", newline='') as file:
                reader = csv.reader(file)
                existing_header = next(reader, None)
                if existing_header is None:
                    raise ValueError("CSV file exists but is empty.")
                self.header = existing_header

            # If header is given, validate match
            if header is not None:
                expected_header = ["Epoch"]+ self.validate_header(header)
                if self.header != expected_header:
                    raise ValueError(f"Header mismatch: expected {expected_header}, found {self.header}")
        else:
            # Create new file with header
            if not isinstance(header, list) or not header:
                raise ValueError("Header must be a non-empty list when creating a new file.")
            self.header = ["Epoch"]+self.validate_header(header)
            with self.filepath.open(mode='a', encoding="utf-8", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.header)


    def log(self, epoch, data):
        # check if the file has been created
        if not self.filepath.is_file():
            raise FileNotFoundError(f"File does not exist: {self.filepath}")
        if not (isinstance(epoch, (int, float)) and epoch >= 0):
            raise ValueError("'epoch' must be a int >= 0")
        
        row = [epoch]
        # if data is a dict, check if all keys are in the header
        if isinstance(data, dict):       
            missing_keys = [key for key in self.header[1:] if key not in data]
            if missing_keys:
                logging.debug(f"Missing keys in data: {missing_keys}")
            for key in data:
                if key not in self.header:
                    raise ValueError(f"Key '{key}' not in header: {self.header}")
            for key in self.header[1:]:
                value = data.get(key, "")
                row.append(self.validate(value) if value != "" else "")
        # if data is a list, check the number of keys
        elif isinstance(data, list):
            if len(data) != len(self.header) - 1:
                raise ValueError(f"Data length {len(data)} does not match header length {len(self.header) - 1}.")
            row += [self.validate(value) for value in data]
        else:
            row.append(self.validate(data))
    
        with open(self.filepath, mode='a', encoding="utf-8", newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(row)