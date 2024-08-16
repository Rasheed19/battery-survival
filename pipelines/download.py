from utils.generic_helper import download_file, get_logger

logger = get_logger(__name__)


def download_pipeline() -> None:
    DESTINATION_FOLDER: str = "data"
    logger.info("Data download pipeline has started.")
    mat_filenames = [
        "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
        "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
        "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
        "2018-08-28_batchdata_updated_struct_errorcorrect.mat",
        "2018-09-02_batchdata_updated_struct_errorcorrect.mat",
        "2018-09-06_batchdata_updated_struct_errorcorrect.mat",
        "2018-09-10_batchdata_updated_struct_errorcorrect.mat",
        "2019-01-24_batchdata_updated_struct_errorcorrect.mat",
    ]
    url_list = [
        "https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download",
        "https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download",
        "https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download",
        "https://data.matr.io/1/api/v1/file/5dcef689110002c7215b2e63/download",
        "https://data.matr.io/1/api/v1/file/5dceef1e110002c7215b28d6/download",
        "https://data.matr.io/1/api/v1/file/5dcef6fb110002c7215b304a/download",
        "https://data.matr.io/1/api/v1/file/5dceefa6110002c7215b2aa9/download",
        "https://data.matr.io/1/api/v1/file/5dcef152110002c7215b2c90/download",
    ]

    for i, (file_name, url) in enumerate(zip(mat_filenames, url_list), start=1):
        print(f"{i} of 8: downloading {file_name} from {url}...")
        download_file(
            url=url, file_name=file_name, destination_folder=DESTINATION_FOLDER
        )

    logger.info(
        "Data download pipeline finished successfuly. "
        f"All data downloaded to {DESTINATION_FOLDER}."
    )
