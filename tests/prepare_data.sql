drop database if exists test_frs;
create database test_frs;
drop database if exists test_lprs;
create database test_lprs;

drop user if exists test_falprs;
create user test_falprs with encrypted password '123';

grant all on database test_frs to test_falprs;
alter database test_frs owner to test_falprs;
grant all on database test_lprs to test_falprs;
alter database test_lprs owner to test_falprs;
