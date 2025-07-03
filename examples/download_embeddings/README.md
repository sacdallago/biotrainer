# download embeddings example

The main goal of this example is to show you how to download your custom embeddings from a URL.
The following protocols are allowed: `http, https, ftp`.
Embeddings are downloaded and saved in the directory of this example with a "downloaded_" prefix.

Execute the example (from the base directory):
```bash
biotrainer train --config examples/download_embeddings/config.yml
```