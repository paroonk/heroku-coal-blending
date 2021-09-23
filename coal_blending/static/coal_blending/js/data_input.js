$(document).ready(function() {
    $('#Table1').DataTable({
        paging: false,
        "searching": false,
        "bInfo" : false,
        'lengthMenu': [[10, 25, 50, -1], [10, 25, 50, "All"]],
    });
    $('#Table2').DataTable({
        paging: false,
        "searching": false,
        "ordering": false,
        "bInfo" : false,
        'lengthMenu': [[10, 25, 50, -1], [10, 25, 50, "All"]],
    });
    $('#Table3').DataTable({
        paging: false,
        "searching": false,
        "ordering": false,
        "bInfo" : false,
        'lengthMenu': [[10, 25, 50, -1], [10, 25, 50, "All"]],
    });
    $('#Table4').DataTable({
        paging: true,
        "searching": false,
        "ordering": false,
        "bInfo" : false,
        'lengthMenu': [[10, 25, 50, -1], [10, 25, 50, "All"]],
    });
    $('#Table5').DataTable({
        paging: true,
        "searching": false,
        "ordering": false,
        "bInfo" : false,
        'lengthMenu': [[10, 25, 50, -1], [10, 25, 50, "All"]],
    });
});
