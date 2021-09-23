$(document).ready(function() {
    $('#dbTable').DataTable({
        paging: false,
        'lengthMenu': [[10, 25, 50, -1], [10, 25, 50, "All"]],
    });
});
