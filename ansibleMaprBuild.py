#
#   ansible-playbook /ansible/rh7/install_mapr_pre1.yml  -i /ansible/rh7/<Your MSid>.ctrl


#   pre-install
install_mapr_pre1.yml

#   fix ssh
/ opt / mapr / uhg_admin / bin / bdp_fix_ssh_group.sh < clush_group >
/ opt / mapr / uhg_admin / bin / bdp_fix_ssh_group.sh newdata

# From dbsld0031 only – as root(sudo su - ):
/ opt / mapr / uhg_admin / bin / bdp_fix_ssh_group.sh < clush_group >
/ opt / mapr / uhg_admin / bin / bdp_fix_ssh_group.sh newdata


#   4.	Validate if host is registered to subscription manager:

clush -g newdata -b "sudo subscription-manager status"
    # what do we need back: "Overall Status: Current" - options/branching?
    if (overall_status=="CURRENT"):
        clush -g newdata -b "sudo subscription-manager list"
        # what is expected output here? What if it is not?
    else:
        # what here?


clush -g newdata -b "sudo subscription-manager identity"
#   what is expected output here?
    if (return!="environment name: 7_7_3x86/RHEL_7_3_x86_64_IMS")
        #   what if it is not here?
        sudo subscription-manager register --org="Optum" --activationkey="IMS 7DEVx86" --force
        # [TBD] what action/check here?
    # then loop back to start of the step?




#   5.	OS Filesystem re-sizing
clush -g newdata sudo vgs //
#   Check if appvg is present, and if present, run the following scripts
    if (appvg.present):
        sudo clush -g newdata "/tmp/bdp_moveto_rootvg"
        sudo clush -g newdata "/tmp/bdp_MultipleFS_extend"


    #   validation:
    Validation:
    clush -g newdata "df -h |grep -i /opt/mapr/tmp"
    clush -g newdata "df -h |grep -i /opt/cores"
    clush -g newdata "df -h |grep -i /opt/sasep"
    clush -g newdata "df -h |grep -i /opt/splunk"

    # expected output - are the numbers exact? If not provide more rules/range/exact programmable rule
    /opt/mapr/tmp  64 GB
    /opt/cores     64G
    /opt/sasep     5.1G
    /opt/splunk    11G



#   6.	Install MapR RPMs:
cd /ansible/rh7
ansible-playbook /ansible/rh7/install_mapr_1.yml -i /ansible/rh7/<Your MSid>.ctrl
#   TBD - Improvement opp: Consider using lineinfile instead of sed for warden conf changes


ansible-playbook /ansible/rh7/install_datascience_tools.yml -i /ansible/rh7/<Your MSid>.ctrl

#   TBD: Frank to test this yml file on one data node and 1 edge node and validate with Rajesh/Binu
#   before running it on all RH7 nodes


#   7.	Start warden on nodes:
sudo clush -g newdata "/usr/bin/systemctl start mapr-warden"    # rewrite into ansible [TBD]
#   what here?: Warning: mapr-warden.service changed on disk. Run 'systemctl daemon-reload' to reload units.

#   Validate if /mapr is mounted
[mapr@dbsld0031:/ansible/rh7] sudo clush -g newdata "df -h | grep local"
    #   TBD - what is epxected output here? TBD - code in ansible


#   NEXT:
#   8.	Add disks to the data node






























