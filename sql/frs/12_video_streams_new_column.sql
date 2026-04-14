alter table video_streams
    add if not exists callback_url_barcodes varchar(200);
comment on column video_streams.callback_url_barcodes is 'URL for barcode callbacks';
