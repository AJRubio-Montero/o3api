#! /usr/bin/expect -f
# Copyright (c) 2018 - 2020 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
# @author: vykozlov
#
# simple script to unlock programmatically an oidc-agent account
# requires: expect
# usage: oidc-unlock account passw


set OACCOUNT [lindex $argv 0]
set OPASSW [lindex $argv 1]
spawn oidc-add $OACCOUNT
expect "Enter encryption password for account config $OACCOUNT:"
send "$OPASSW\r"
interact
